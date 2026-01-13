"""
train_gcn.py

Train a *retrieval-aligned* molecule encoder that maps molecular graphs into the
same metric space as text embeddings.

Key upgrades vs a naive baseline:
- Use *categorical embeddings per feature column* for atoms and bonds
- Use an *edge-aware* message passing layer (GINEConv) that conditions on bond features
- Train with a *symmetric contrastive (CLIP-style) loss* instead of MSE regression
- Use *mean pooling* + projection + L2 normalization for stable cosine retrieval
- Optional *node-count capped batching* for variable-size graphs

This still follows the "graph -> embedding -> retrieve nearest caption" philosophy.
"""

from __future__ import annotations

import argparse
import os
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.nn import GINEConv, global_mean_pool

# Imports from baseline folder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from data_utils import (
    FeatureCardinalities,
    PreprocessedGraphDataset,
    NodeCountBatchSampler,
    collate_fn,
    infer_feature_cardinalities,
    load_graphs,
    # load_id2emb,
)

from data_baseline.data_utils import load_id2emb

# -------------------------
# Default paths (override via CLI)
# -------------------------
DATA_DIR = "data"
TRAIN_GRAPHS = os.path.join(DATA_DIR, "train_graphs.pkl")
VAL_GRAPHS = os.path.join(DATA_DIR, "validation_graphs.pkl")
TEST_GRAPHS = os.path.join(DATA_DIR, "test_graphs.pkl")
TRAIN_EMB_CSV = os.path.join(DATA_DIR, "train_text_embeddings.csv")
VAL_EMB_CSV = os.path.join(DATA_DIR, "validation_text_embeddings.csv")

# Select device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# -------------------------
# Model
# -------------------------
class ColumnwiseCategoricalEmbedding(nn.Module):
    """
    Embed each categorical feature column with its own embedding table, then combine.

    Combine strategy: sum of per-column embeddings (simple + strong).
    """

    def __init__(self, cardinalities: List[int], emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(c), emb_dim) for c in cardinalities]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C] integer tensor
        returns: [N, emb_dim]
        """
        if x is None:
            raise ValueError("Expected x tensor for categorical embedding")
        if x.dtype != torch.long:
            x = x.long()
        # sum embeddings over columns
        out = 0.0
        for col, emb in enumerate(self.embeddings):
            out = out + emb(x[:, col])
        return self.dropout(out)


class MolGNN(nn.Module):
    """
    Edge-aware GNN encoder (GINE) for molecular graphs.
    Produces a normalized graph embedding.
    """

    def __init__(
        self,
        feat_cards: FeatureCardinalities,
        text_in_dim: int,
        model_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = dict(
            node_card=feat_cards.node,
            edge_card=feat_cards.edge,
            text_in_dim=text_in_dim,
            model_dim=model_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.node_emb = ColumnwiseCategoricalEmbedding(
            feat_cards.node, model_dim, dropout=0.0
        )
        self.edge_emb = ColumnwiseCategoricalEmbedding(
            feat_cards.edge, model_dim, dropout=0.0
        )

        # GINEConv expects edge_attr with same dim as node dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=model_dim))
            self.norms.append(nn.LayerNorm(model_dim))

        self.dropout = nn.Dropout(dropout)
        self.graph_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, out_dim),
        )

        # Text projection head: maps fixed text embeddings to out_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_in_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, out_dim),
        )

        # CLIP-style learnable logit scale (temperature)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))  # exp(logit_scale) / tau

    @torch.no_grad()
    def encode_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        z = self.text_proj(text_emb)
        return F.normalize(z, dim=-1)

    def encode_graph(self, batch) -> torch.Tensor:
        """
        batch: PyG Batch with .x, .edge_index, .edge_attr, .batch
        returns: [B, out_dim] normalized
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)

        # Sanity check
        x = batch.x.long()
        cards = self.cfg["node_card"]  # list of K per column
        for col, K in enumerate(cards):
            mn = int(x[:, col].min())
            mx = int(x[:, col].max())
            if mn < 0 or mx >= K:
                bad = x[(x[:, col] < 0) | (x[:, col] >= K), col][:10].tolist()
                raise RuntimeError(
                    f"Node feature out of range in col {col}: min={mn}, max={mx}, K={K}, examples={bad}"
                )

        # Embed
        h = self.node_emb(x)

        if edge_attr is None:
            # If edge_attr is missing, fall back to zeros (still works but suboptimal)
            E = edge_index.size(1)
            edge_attr = torch.zeros(
                (E, len(self.edge_emb.embeddings)), device=h.device, dtype=torch.long
            )
        e = self.edge_emb(edge_attr)

        for conv, ln in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, e)
            h_new = ln(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual

        g = global_mean_pool(h, batch.batch)
        g = self.graph_proj(g)
        g = F.normalize(g, dim=-1)
        return g

    def forward(
        self, batch, text_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_graph(batch), self.encode_text(text_emb)


# -------------------------
# Loss + metrics
# -------------------------
def clip_contrastive_loss(
    g: torch.Tensor,
    t: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric CLIP-style contrastive loss with in-batch negatives.

    g: [B, D] normalized
    t: [B, D] normalized
    """
    # Clamp logit scale to avoid overflow (like OpenAI CLIP)
    scale = logit_scale.exp().clamp(min=1e-3, max=100.0)
    logits = scale * (g @ t.T)  # [B, B]
    labels = torch.arange(g.size(0), device=g.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i + loss_t)


@torch.no_grad()
def retrieval_metrics(
    g: torch.Tensor, t: torch.Tensor, ks=(1, 5, 10)
) -> Dict[str, float]:
    """
    Retrieval among paired sets (same size): graph->text and text->graph.

    g: [N, D] normalized
    t: [N, D] normalized
    """
    sim = g @ t.T  # cosine since normalized

    # Graph -> Text ranks
    order_g2t = sim.argsort(dim=1, descending=True)  # [N, N]
    # True at (i, rank_position_of_correct_text_for_i)
    hits_g2t = (
        order_g2t == torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    ).nonzero()
    # hits_g2t rows: [i, rank]
    hits_g2t = hits_g2t[hits_g2t[:, 0].argsort()]  # sort by i
    ranks_g2t = hits_g2t[:, 1]

    # Text -> Graph ranks
    order_t2g = sim.argsort(
        dim=0, descending=True
    )  # [N, N], order_t2g[:, j] is ranked graph indices for text j
    hits_t2g = (
        order_t2g == torch.arange(sim.size(0), device=sim.device).unsqueeze(0)
    ).nonzero()
    # hits_t2g rows: [rank, j]
    hits_t2g = hits_t2g[hits_t2g[:, 1].argsort()]  # sort by j
    ranks_t2g = hits_t2g[:, 0]

    out: Dict[str, float] = {}
    for k in ks:
        out[f"R@{k}_g2t"] = float((ranks_g2t < k).float().mean().item())
        # out[f"R@{k}_t2g"] = float((ranks_t2g < k).float().mean().item())  # COMMENTED OUT T2G
    out["MRR_g2t"] = float((1.0 / (ranks_g2t.float() + 1.0)).mean().item())
    # out["MRR_t2g"] = float((1.0 / (ranks_t2g.float() + 1.0)).mean().item())
    return out


# -------------------------
# Train / eval loops
# -------------------------
def train_one_epoch(
    model: MolGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()  # Train mode
    total_loss = 0.0
    n = 0

    for batch_graph, text_emb, _ids in loader:
        batch_graph = batch_graph.to(device)
        text_emb = text_emb.to(device)

        g, t = model(batch_graph, text_emb)
        loss = clip_contrastive_loss(g, t, model.logit_scale)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * g.size(0)
        n += g.size(0)

    return total_loss / max(1, n)


@torch.no_grad()
def eval_val_split(
    model: MolGNN,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    g_all = []
    t_all = []
    for batch_graph, text_emb, _ids in val_loader:
        batch_graph = batch_graph.to(device)
        text_emb = text_emb.to(device)
        g = model.encode_graph(batch_graph)
        t = model.encode_text(text_emb)
        g_all.append(g)
        t_all.append(t)
    g_all = torch.cat(g_all, dim=0)
    t_all = torch.cat(t_all, dim=0)
    return retrieval_metrics(g_all, t_all)


def save_checkpoint(path: str, model: MolGNN, extra: Optional[Dict] = None):
    """Save a self-describing checkpoint.

    We keep it lightweight (model weights + model config), but allow optional
    extra metadata (epoch, best metric, args, optimizer, scheduler, ...).
    """
    payload = {
        "state_dict": model.state_dict(),
        "config": model.cfg,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> MolGNN:
    payload = torch.load(path, map_location=device)
    cfg = payload["config"]
    feat_cards = FeatureCardinalities(node=cfg["node_card"], edge=cfg["edge_card"])
    model = MolGNN(
        feat_cards=feat_cards,
        text_in_dim=int(cfg["text_in_dim"]),
        model_dim=int(cfg["model_dim"]),
        out_dim=int(cfg["out_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_graphs", default="data/train_graphs.pkl")
    parser.add_argument("--val_graphs", default="data/validation_graphs.pkl")
    parser.add_argument("--train_emb_csv", default="data/train_text_embeddings.csv")
    parser.add_argument("--val_emb_csv", default="data/validation_text_embeddings.csv")
    parser.add_argument("--out", default="checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Used only if max_nodes_per_batch is 0.",
    )
    parser.add_argument(
        "--max_nodes_per_batch",
        type=int,
        default=4000,
        help="0 disables node-capped batching.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # Training control: LR schedule + early stopping (set-and-forget)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_warmup",
        choices=["none", "cosine", "cosine_warmup"],
        help="Learning-rate schedule. cosine_warmup is usually best for contrastive training.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum LR reached at the end of training for cosine schedules.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="Warmup epochs for cosine_warmup schedule.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=8,
        help="Stop if MRR_g2t doesn't improve for this many validation epochs. 0 disables.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum absolute improvement in MRR_g2t to reset patience.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run quick mock run with data/*_mock.pkl and 1 epoch to test the pipeline.",
    )

    # Parse config
    parser.add_argument("--config", type=str, default=None)

    # Pre-parse only to get --config
    cfg_args, _ = parser.parse_known_args()

    cfg = {}
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

        # optional safety check
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(
                f"Unknown config keys in {cfg_args.config}: {sorted(unknown)}"
            )

        parser.set_defaults(**cfg)

    # Parse all args
    args = parser.parse_args()
    args.config = cfg_args.config

    # ------------------------------------------------------------------
    # MOCK MODE: override paths and params for a small dry run
    # ------------------------------------------------------------------
    if args.mock:
        print("\n🧪 MOCK MODE ENABLED — running short pipeline test\n")
        args.train_graphs = "data/train_graphs_mock.pkl"
        args.val_graphs = "data/validation_graphs_mock.pkl"
        args.train_emb_csv = "data/train_text_embeddings.csv"
        args.val_emb_csv = "data/validation_text_embeddings.csv"
        args.epochs = 1
        args.batch_size = 8
        args.out = "tmp/mock_ckpt.pt"

        # auto-create tmp folder if needed
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("\n========== RUN CONFIG ==========")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("================================\n")

    # Fix seed
    torch.manual_seed(args.seed)

    # Load embeddings
    train_id2emb = load_id2emb(args.train_emb_csv)
    val_id2emb = load_id2emb(args.val_emb_csv)

    # infer text embedding dimension from first element
    text_dim = next(iter(train_id2emb.values())).shape[0]

    # ---------------------------------------------------------
    # Load graphs for cardinalities (train + val to avoid OOB in mock)
    # ---------------------------------------------------------
    train_graphs = load_graphs(args.train_graphs)

    val_graphs = []
    if os.path.exists(args.val_graphs):
        val_graphs = load_graphs(args.val_graphs)

    feat_cards = infer_feature_cardinalities(train_graphs + val_graphs)

    # Datasets
    train_ds = PreprocessedGraphDataset(args.train_graphs, id2emb=train_id2emb)
    if val_id2emb is not None and os.path.exists(args.val_graphs):
        val_ds = PreprocessedGraphDataset(args.val_graphs, id2emb=val_id2emb)
    else:
        val_ds = None

    # Loaders (size-aware batching recommended)
    if args.max_nodes_per_batch and args.max_nodes_per_batch > 0:
        train_sampler = NodeCountBatchSampler(
            train_ds, max_nodes=args.max_nodes_per_batch, shuffle=True
        )
        train_dl = DataLoader(
            train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

    if val_ds is not None:
        # deterministic batching for eval
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
    else:
        val_dl = None

    model = MolGNN(
        feat_cards=feat_cards,
        text_in_dim=text_dim,
        model_dim=args.model_dim,
        out_dim=args.out_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # -------------------------
    # LR scheduler (epoch-wise)
    # -------------------------
    scheduler = None
    if args.scheduler != "none":
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.min_lr
            )
        elif args.scheduler == "cosine_warmup":
            warm = max(0, int(args.warmup_epochs))
            total = max(1, int(args.epochs))
            # ratio of final LR to initial LR, clamped to [0, 1]
            min_ratio = float(args.min_lr) / float(args.lr)
            min_ratio = min(max(min_ratio, 0.0), 1.0)

            def lr_lambda(epoch: int) -> float:
                # Linear warmup: 0 -> 1 over warm epochs
                if warm > 0 and epoch < warm:
                    return float(epoch + 1) / float(warm)
                # Cosine decay: 1 -> min_ratio
                if total - warm <= 0:
                    return 1.0
                progress = float(epoch - warm) / float(total - warm)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )

    print(f"Device: {DEVICE}")
    print(f"Train graphs: {len(train_ds)} | Text dim: {text_dim}")
    print(f"Node cards: {feat_cards.node} | Edge cards: {feat_cards.edge}")
    if val_dl is not None:
        print(f"Val graphs: {len(val_ds)}")
    print()

    # Save two checkpoints: best (selected by MRR_g2t) and last (final weights)
    best_path = args.out
    last_path = (
        args.out[:-3] + "_last.pt"
        if args.out.endswith(".pt")
        else args.out + "_last.pt"
    )

    best_mrr = None
    bad_epochs = 0

    for ep in range(args.epochs):
        tr_loss = train_one_epoch(
            model, train_dl, optimizer, DEVICE, grad_clip=args.grad_clip
        )
        lr_now = optimizer.param_groups[0]["lr"]
        msg = f"Epoch {ep+1:03d}/{args.epochs}  loss={tr_loss:.4f}  lr={lr_now:.2e}"
        if val_dl is not None:
            scores = eval_val_split(model, val_dl, DEVICE)
            msg += "  " + " ".join([f"{k}={v:.4f}" for k, v in scores.items()])
            # best tracking + early stopping on MRR_g2t
            key = float(scores.get("MRR_g2t", 0.0))
            improved = (best_mrr is None) or (
                key > best_mrr + args.early_stop_min_delta
            )
            if improved:
                best_mrr = key
                bad_epochs = 0
                save_checkpoint(
                    best_path,
                    model,
                    extra={
                        "epoch": ep + 1,
                        "best_mrr_g2t": best_mrr,
                        "train_args": vars(args),
                    },
                )
                msg += "  [saved_best]"
            else:
                bad_epochs += 1

            # Early stop if no progress for `patience` epochs
            if args.early_stop_patience and args.early_stop_patience > 0:
                if bad_epochs >= args.early_stop_patience:
                    msg += f"  [early_stop: {bad_epochs} epochs w/o MRR_g2t improv]"
                    # still write last checkpoint before stopping
                    save_checkpoint(
                        last_path,
                        model,
                        extra={
                            "epoch": ep + 1,
                            "best_mrr_g2t": best_mrr,
                            "train_args": vars(args),
                        },
                    )
                    print(msg)
                    break
        else:
            # still save last
            save_checkpoint(
                last_path,
                model,
                extra={"epoch": ep + 1, "train_args": vars(args)},
            )

        # Always keep a "last" checkpoint (handy for debugging/resume)
        if val_dl is not None:
            save_checkpoint(
                last_path,
                model,
                extra={
                    "epoch": ep + 1,
                    "best_mrr_g2t": best_mrr,
                    "train_args": vars(args),
                },
            )

        # Step scheduler at end of epoch
        if scheduler is not None:
            scheduler.step()
        print(msg)

    # Ensure we have a final last checkpoint even if loop ended normally
    if not os.path.exists(last_path):
        save_checkpoint(
            last_path,
            model,
            extra={
                "epoch": args.epochs,
                "best_mrr_g2t": best_mrr,
                "train_args": vars(args),
            },
        )

    if val_dl is None:
        print(f"\nCheckpoint written to: {last_path}")
    else:
        print(f"\nBest checkpoint: {best_path}")
        print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
