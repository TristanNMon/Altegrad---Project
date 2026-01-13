# Retrieval-Aligned GNN Baseline (Upgraded)

This `train_gcn.py` trains a **molecule graph encoder** whose output lives in the **same cosine-similarity space as text embeddings**, so you can retrieve the nearest training caption for a test molecule (“graph → embedding → nearest caption”).

It is a drop-in conceptual upgrade of the original baseline (GCN + MSE), but now the model is **chemistry-aware** and the training objective is **retrieval-aligned**.

---

## What changed vs the original baseline (and why)

### 1) Atom & bond features are now used (categorical column embeddings)

**Baseline:** every node starts from the *same* learned vector → the model mainly learns from graph topology.
**Upgrade:** each of the 9 atom feature columns and 3 bond feature columns gets its own embedding table, then embeddings are combined per node/edge.

**Why it matters:** molecular captions depend heavily on **atom identity, aromaticity, ring membership, charge, bond type, stereo, conjugation**—all of which were effectively ignored before.

---

### 2) Message passing is now edge-aware (GINEConv)

**Baseline:** `GCNConv` ignores `edge_attr`, so “single vs double vs aromatic” is invisible.
**Upgrade:** `GINEConv` conditions messages on embedded bond features.

**Why it matters:** bond semantics (type/stereo/conjugation) are core to functional groups and naming; this directly improves discriminability for retrieval.

---

### 3) Training objective is retrieval-first (CLIP-style contrastive loss)

**Baseline:** MSE regression to match the paired text embedding, with no explicit “push away wrong captions.”
**Upgrade:** symmetric contrastive loss with **in-batch negatives** (graph→text and text→graph).

**Why it matters:** retrieval is a **ranking** problem. Contrastive learning teaches “match your caption *and be dissimilar to others*,” which usually improves Recall@K/MRR much more than MSE.

---

### 4) Pooling and projection are stabilized for cosine retrieval

**Baseline:** sum pooling (`global_add_pool`) + projection + normalize.
**Upgrade:** mean pooling (`global_mean_pool`) + projection head + **L2-normalization**.

**Why it matters:** mean pooling reduces unintended dependence on graph size; normalized outputs make cosine similarity stable and comparable across batches.

---

### 5) Better training hygiene for variable-size graphs

**Upgrade adds:**

* **Node-count capped batching** (optional): batches are formed so total nodes ≤ `max_nodes_per_batch`
* gradient clipping
* cosine LR schedules (with optional warmup)
* early stopping on **MRR_g2t**
* “mock mode” for quick pipeline tests

**Why it matters:** molecule graphs vary widely in size; size-aware batching + stable optimization prevents outliers from dominating memory and gradients.

---

## Inputs / outputs

### Required inputs

* `data/train_graphs.pkl`, `data/validation_graphs.pkl` (PyG `Data` objects)
* `data/train_text_embeddings.csv`, `data/validation_text_embeddings.csv` (fixed text embeddings keyed by id)

### Output checkpoints

Two checkpoints are saved:

* **best** (selected by validation `MRR_g2t`) → `--out checkpoint.pt`
* **last** (final epoch weights) → `checkpoint_last.pt`

Checkpoints are **self-describing**: they store model weights **and** the model configuration needed to reload the architecture.

---

## What metric the training loop optimizes

During validation, it computes retrieval metrics in the shared embedding space:

* `R@1`, `R@5`, `R@10` for **graph→text**
* `MRR_g2t` (mean reciprocal rank)

`MRR_g2t` is used for:

* best checkpoint selection
* early stopping
* (optionally) ReduceLROnPlateau scheduling
