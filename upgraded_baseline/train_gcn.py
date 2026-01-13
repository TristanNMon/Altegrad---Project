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
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


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
        out[f"R@{k}_t2g"] = float((ranks_t2g < k).float().mean().item())
    out["MRR_g2t"] = float((1.0 / (ranks_g2t.float() + 1.0)).mean().item())
    out["MRR_t2g"] = float((1.0 / (ranks_t2g.float() + 1.0)).mean().item())
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


def save_checkpoint(path: str, model: MolGNN):
    payload = {
        "state_dict": model.state_dict(),
        "config": model.cfg,
    }
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
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run quick mock run with data/*_mock.pkl and 1 epoch to test the pipeline.",
    )

    args = parser.parse_args()

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

    # Fix seed
    torch.manual_seed(args.seed)

    # Load embeddings
    train_id2emb = load_id2emb(args.train_emb_csv)
    val_id2emb = load_id2emb(args.val_emb_csv)

    # infer text embedding dimension from first element
    text_dim = next(iter(train_id2emb.values())).shape[0]

    # Load graphs for cardinalities (use train graphs only)
    train_graphs = load_graphs(args.train_graphs)
    feat_cards = infer_feature_cardinalities(train_graphs)

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

    print(f"Device: {DEVICE}")
    print(f"Train graphs: {len(train_ds)} | Text dim: {text_dim}")
    print(f"Node cards: {feat_cards.node} | Edge cards: {feat_cards.edge}")
    if val_dl is not None:
        print(f"Val graphs: {len(val_ds)}")
    print()

    best = None
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(
            model, train_dl, optimizer, DEVICE, grad_clip=args.grad_clip
        )
        msg = f"Epoch {ep+1:03d}/{args.epochs}  loss={tr_loss:.4f}"
        if val_dl is not None:
            scores = eval_val_split(model, val_dl, DEVICE)
            msg += "  " + " ".join([f"{k}={v:.4f}" for k, v in scores.items()])
            # naive best tracking on MRR_g2t
            key = scores.get("MRR_g2t", 0.0)
            if best is None or key > best:
                best = key
                save_checkpoint(args.out, model)
                msg += "  [saved]"
        else:
            # still save last
            save_checkpoint(args.out, model)
        print(msg)

    if val_dl is None:
        save_checkpoint(args.out, model)
    print(f"\nCheckpoint written to: {args.out}")


if __name__ == "__main__":
    main()
