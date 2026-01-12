"""
data_utils.py

Utilities for the molecule ↔ text *retrieval* baseline.

The dataset provides:
- Pickled PyG `Data` objects for graphs (train/val/test).
- A CSV mapping graph ids -> text embedding vectors (train, optionally val).

This file focuses on:
- Robust loading of graph pickles and embedding CSVs
- PyTorch Dataset + collate utilities
- Small helpers for feature-cardinality inference (for categorical embeddings)

Assumptions (based on the challenge preprocessing):
- Each graph has `id` (string or int), stored as `graph.id`
- Node features are categorical ints in `graph.x` with shape [num_nodes, 9]
- Edge indices in `graph.edge_index` with shape [2, num_edges]
- Edge features are categorical ints in `graph.edge_attr` with shape [num_edges, 3]
- Train/val graphs may have `description` (string). Test graphs do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import ast
import os
import math
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch


Tensor = torch.Tensor


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------
def load_graphs(pkl_path: str) -> List[object]:
    """Load a list of PyG Data objects from a pickle file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Graph pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)
    if not isinstance(data_list, list):
        raise ValueError(f"Expected a list in {pkl_path}, got: {type(data_list)}")
    return data_list


def _parse_embedding_cell(cell) -> Optional[List[float]]:
    """Parse common CSV embedding representations into a list[float]."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None

    # Already a list/tuple
    if isinstance(cell, (list, tuple)):
        return [float(x) for x in cell]

    # String encodings: "[0.1, 0.2]" or "0.1 0.2 0.3" etc.
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return None
        # Try python literal list
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [float(x) for x in obj]
        except Exception:
            pass
        # Try split by comma or whitespace
        if "," in s:
            parts = [p for p in s.split(",") if p.strip()]
        else:
            parts = [p for p in s.split() if p.strip()]
        try:
            return [float(p) for p in parts]
        except Exception:
            return None

    return None


# def load_descriptions_from_graphs(pkl_path: str) -> Dict[str, str]:
#     """Return {graph_id: description} for graphs that have a non-empty 'description'."""
#     graphs = load_graphs(pkl_path)
#     out: Dict[str, str] = {}
#     for g in graphs:
#         gid = str(getattr(g, "id", ""))
#         desc = getattr(g, "description", None)
#         if gid and isinstance(desc, str) and desc.strip():
#             out[gid] = desc
#     return out


# ---------------------------------------------------------------------
# Feature-cardinality inference (for categorical embeddings)
# ---------------------------------------------------------------------
@dataclass
class FeatureCardinalities:
    node: List[int]  # length = num_node_cols (e.g., 9)
    edge: List[int]  # length = num_edge_cols (e.g., 3)


def infer_feature_cardinalities(graphs: Sequence[object]) -> FeatureCardinalities:
    """
    Infer embedding table sizes as (max_value + 1) per categorical column.
    """
    node_max: Optional[List[int]] = None
    edge_max: Optional[List[int]] = None

    for g in graphs:
        x = getattr(g, "x", None)
        if (
            x is not None
            and hasattr(x, "shape")
            and len(x.shape) == 2
            and x.shape[0] > 0
        ):
            x_cpu = x.detach().cpu() if hasattr(x, "detach") else x
            # ensure integer
            x_cpu = (
                x_cpu.to(torch.long)
                if hasattr(x_cpu, "to")
                else torch.tensor(x_cpu, dtype=torch.long)
            )
            col_max = x_cpu.max(dim=0).values.tolist()
            if node_max is None:
                node_max = [int(v) for v in col_max]
            else:
                node_max = [max(a, int(b)) for a, b in zip(node_max, col_max)]

        ea = getattr(g, "edge_attr", None)
        if (
            ea is not None
            and hasattr(ea, "shape")
            and len(ea.shape) == 2
            and ea.shape[0] > 0
        ):
            ea_cpu = ea.detach().cpu() if hasattr(ea, "detach") else ea
            ea_cpu = (
                ea_cpu.to(torch.long)
                if hasattr(ea_cpu, "to")
                else torch.tensor(ea_cpu, dtype=torch.long)
            )
            col_max = ea_cpu.max(dim=0).values.tolist()
            if edge_max is None:
                edge_max = [int(v) for v in col_max]
            else:
                edge_max = [max(a, int(b)) for a, b in zip(edge_max, col_max)]

    if node_max is None:
        # fallback to 9 columns if absent
        node_max = [0] * 9
    if edge_max is None:
        edge_max = [0] * 3

    node_card = [m + 1 for m in node_max]
    edge_card = [m + 1 for m in edge_max]
    return FeatureCardinalities(node=node_card, edge=edge_card)


# ---------------------------------------------------------------------
# Dataset + batching
# ---------------------------------------------------------------------
class PreprocessedGraphDataset(Dataset):
    """
    Dataset over pickled graphs.

    Modes:
      - If id2emb is provided: returns (graph, text_embedding, graph_id)
      - Else: returns (graph, graph_id)

    Notes:
      - We keep graph objects intact (PyG Data).
      - We also expose num_nodes per item for size-aware batching.
    """

    def __init__(self, graph_pkl: str, id2emb: Optional[Dict[str, Tensor]] = None):
        super().__init__()
        self.graphs = load_graphs(graph_pkl)
        self.id2emb = id2emb

        self.ids: List[str] = []
        self.num_nodes: List[int] = []
        self._keep: List[int] = []

        for idx, g in enumerate(self.graphs):
            gid = str(getattr(g, "id", ""))
            if not gid:
                continue
            if id2emb is not None and gid not in id2emb:
                # If training uses embeddings, skip graphs without embedding.
                continue
            self._keep.append(idx)
            self.ids.append(gid)
            x = getattr(g, "x", None)
            n = int(x.shape[0]) if x is not None and hasattr(x, "shape") else 0
            self.num_nodes.append(n)

    def __len__(self) -> int:
        return len(self._keep)

    def __getitem__(self, i: int):
        g = self.graphs[self._keep[i]]
        gid = self.ids[i]
        if self.id2emb is None:
            return g, gid
        return g, self.id2emb[gid], gid


def collate_fn(batch):
    """
    Collate for PyG graphs + optional text embeddings.

    Returns:
      - if dataset yields (graph, gid): (Batch, gids)
      - if dataset yields (graph, emb, gid): (Batch, embs, gids)
    """
    if len(batch) == 0:
        raise ValueError("Empty batch in collate_fn")

    if len(batch[0]) == 2:
        graphs, gids = zip(*batch)
        return Batch.from_data_list(list(graphs)), list(gids)

    graphs, embs, gids = zip(*batch)
    return (
        Batch.from_data_list(list(graphs)),
        torch.stack(list(embs), dim=0),
        list(gids),
    )


class NodeCountBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that limits the *total number of nodes* in each batch.

    This is a practical way to prevent occasional giant graphs from causing OOM.
    It creates variable-size batches.

    Example:
        sampler = NodeCountBatchSampler(dataset, max_nodes=4000, shuffle=True)
        dl = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    """

    def __init__(
        self, dataset: PreprocessedGraphDataset, max_nodes: int, shuffle: bool = True
    ):
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        self.dataset = dataset
        self.max_nodes = int(max_nodes)
        self.shuffle = shuffle

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            # torch.randperm for determinism with torch.manual_seed
            perm = torch.randperm(len(idxs)).tolist()
            idxs = [idxs[i] for i in perm]

        batch: List[int] = []
        budget = 0
        for idx in idxs:
            n = self.dataset.num_nodes[idx]
            # If one graph is huge, yield it alone.
            if batch and (budget + n) > self.max_nodes:
                yield batch
                batch = []
                budget = 0
            batch.append(idx)
            budget += n
        if batch:
            yield batch

    def __len__(self):
        # rough estimate (not used for correctness)
        if len(self.dataset) == 0:
            return 0
        total_nodes = sum(self.dataset.num_nodes)
        return max(1, math.ceil(total_nodes / self.max_nodes))  # type: ignore[name-defined]
