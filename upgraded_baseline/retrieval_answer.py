"""
retrieval_answer.py

Generate the submission CSV by *retrieving* a caption from the training set.

Pipeline:
1) Load trained checkpoint (graph encoder + text projection head).
2) Project + normalize all training text embeddings once.
3) For each test graph, compute graph embedding and retrieve nearest training caption.

Simple inference upgrade over top-1 NN:
- retrieve top-k, compute a soft prototype (weighted average), then select the best
  candidate among those top-k ("kNN smoothing"). This reduces brittleness.

Output format (typical):
  ID, description
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Imports from baseline folder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    # load_descriptions_from_graphs,  # use baseline version instead
    # load_id2emb, # use baseline version instead
)

from train_gcn import (
    DEVICE,
    TRAIN_GRAPHS,
    TEST_GRAPHS,
    TRAIN_EMB_CSV,
    load_checkpoint,
)

from data_baseline.data_utils import load_id2emb, load_descriptions_from_graphs


@torch.no_grad()
def build_train_corpus(
    model,
    train_graphs_pkl: str,
    train_emb_csv: str,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor, List[str]]:
    """
    Returns:
      train_ids: list of ids aligned with train_text_z
      train_text_z: [N, D] normalized projected text embeddings on `device`
      train_descs: list of descriptions aligned with train_ids
    """
    id2desc = load_descriptions_from_graphs(train_graphs_pkl)
    id2emb = load_id2emb(train_emb_csv)

    # intersection (only ids that have both embedding and description)
    ids = sorted(set(id2desc.keys()).intersection(id2emb.keys()))
    if not ids:
        raise RuntimeError("No training ids with both embeddings and descriptions.")

    text = torch.stack([id2emb[i] for i in ids], dim=0).to(device)
    text_z = model.encode_text(text)  # normalized

    descs = [id2desc[i] for i in ids]

    print(repr(descs[1]))  # debug

    return ids, text_z, descs


@torch.no_grad()
def retrieve_batch(
    g: torch.Tensor,
    train_text_z: torch.Tensor,
    train_descs: List[str],
    k: int = 5,
    smooth: bool = True,
    smooth_temp: float = 0.07,
) -> List[str]:
    """
    g: [B, D] normalized graph embeddings
    train_text_z: [N, D] normalized text embeddings
    Returns list of retrieved descriptions for each query in batch.
    """
    sim = g @ train_text_z.T  # [B, N]

    if k <= 1 or not smooth:
        top1 = sim.argmax(dim=1).tolist()
        return [train_descs[i] for i in top1]

    topk_val, topk_idx = torch.topk(sim, k=k, dim=1)

    # soft prototype in text space
    w = F.softmax(topk_val / smooth_temp, dim=1)  # [B, k]
    # [B, D]
    proto = (w.unsqueeze(-1) * train_text_z[topk_idx]).sum(dim=1)
    proto = F.normalize(proto, dim=-1)

    # choose best among top-k against prototype (stays within retrieved candidates)
    # score [B, k]
    score = (proto.unsqueeze(1) * train_text_z[topk_idx]).sum(dim=-1)
    best_in_topk = score.argmax(dim=1)
    chosen = topk_idx[torch.arange(g.size(0), device=g.device), best_in_topk].tolist()
    return [train_descs[i] for i in chosen]


@torch.no_grad()
def generate_submission(
    checkpoint_path: str,
    train_graphs_pkl: str,
    train_emb_csv: str,
    test_graphs_pkl: str,
    output_csv: str,
    batch_size: int = 64,
    topk: int = 5,
    smooth: bool = True,
) -> None:
    model = load_checkpoint(checkpoint_path, DEVICE).to(DEVICE)
    model.eval()  # Eval mode

    train_ids, train_text_z, train_descs = build_train_corpus(
        model,
        train_graphs_pkl=train_graphs_pkl,
        train_emb_csv=train_emb_csv,
        device=DEVICE,
    )

    test_ds = PreprocessedGraphDataset(test_graphs_pkl, id2emb=None)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    rows = []
    for batch_graph, gids in test_dl:
        batch_graph = batch_graph.to(DEVICE)
        g = model.encode_graph(batch_graph)  # normalized
        preds = retrieve_batch(g, train_text_z, train_descs, k=topk, smooth=smooth)
        rows.extend(list(zip(gids, preds)))

    sub = pd.DataFrame(rows, columns=["ID", "description"])
    sub.to_csv(output_csv, index=False)
    print(f"Saved submission to: {output_csv}  (rows={len(sub)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoint.pt")
    parser.add_argument("--train_graphs", default=TRAIN_GRAPHS)
    parser.add_argument("--train_emb_csv", default=TRAIN_EMB_CSV)
    parser.add_argument("--test_graphs", default=TEST_GRAPHS)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run a quick end-to-end retrieval smoke test using data/*_mock.pkl and tmp/mock_ckpt.pt. "
            "Overrides --ckpt/--train_graphs/--test_graphs/--out."
        ),
    )
    parser.add_argument(
        "--no_smooth", action="store_true", help="Disable kNN smoothing; use top-1."
    )
    args = parser.parse_args()

    if args.mock:
        print("\n🧪 MOCK MODE ENABLED — running short retrieval pipeline test\n")
        args.ckpt = "tmp/mock_ckpt.pt"
        args.train_graphs = "data/train_graphs_mock.pkl"
        args.test_graphs = "data/test_graphs_mock.pkl"
        args.out = "tmp/mock_submission.csv"
        # keep train_emb_csv unchanged by default; it can point to full embeddings.
        args.batch_size = min(args.batch_size, 32)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    generate_submission(
        checkpoint_path=args.ckpt,
        train_graphs_pkl=args.train_graphs,
        train_emb_csv=args.train_emb_csv,
        test_graphs_pkl=args.test_graphs,
        output_csv=args.out,
        batch_size=args.batch_size,
        topk=args.topk,
        smooth=(not args.no_smooth),
    )


if __name__ == "__main__":
    main()
