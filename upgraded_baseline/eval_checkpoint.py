"""
eval_checkpoint.py

Load a trained checkpoint (.pt) from train_gcn.py and evaluate it on a split
(typically validation) without any retraining.

Usage examples:
  python eval_checkpoint.py --ckpt tmp/best.pt
  python eval_checkpoint.py --ckpt tmp/best.pt --val_graphs data/validation_graphs.pkl --val_emb_csv data/validation_text_embeddings.csv
  python eval_checkpoint.py --ckpt tmp/best.pt --mock
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

# Imports from baseline folder (same trick as train script)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
)

from data_baseline.data_utils import load_id2emb

# Import model + eval functions from your training script
from train_gcn import (
    DEVICE,
    load_checkpoint,
    eval_val_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument("--val_graphs", default="data/validation_graphs.pkl")
    parser.add_argument("--val_emb_csv", default="data/validation_text_embeddings.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Evaluate on mock validation split (data/validation_graphs_mock.pkl).",
    )
    args = parser.parse_args()

    if args.mock:
        args.val_graphs = "data/validation_graphs_mock.pkl"
        args.val_emb_csv = "data/validation_text_embeddings.csv"
        args.batch_size = min(args.batch_size, 32)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.exists(args.val_graphs):
        raise FileNotFoundError(f"Validation graphs not found: {args.val_graphs}")
    if not os.path.exists(args.val_emb_csv):
        raise FileNotFoundError(
            f"Validation embeddings CSV not found: {args.val_emb_csv}"
        )

    # Load validation embeddings + dataset
    val_id2emb = load_id2emb(args.val_emb_csv)
    val_ds = PreprocessedGraphDataset(args.val_graphs, id2emb=val_id2emb)

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load model checkpoint (no training)
    model = load_checkpoint(args.ckpt, DEVICE).to(DEVICE)
    model.eval()

    # Compute retrieval metrics
    scores = eval_val_split(model, val_dl, DEVICE)

    print(f"\nDevice: {DEVICE}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Val graphs: {len(val_ds)}")
    print("Metrics:")
    for k, v in sorted(scores.items()):
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
