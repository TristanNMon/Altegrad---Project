# ALTEGRAD 2025 Data Challenge: Molecular Graph Captioning

**Authors:** Tristan Montalbetti & Jules Royer  
**Date:** January 2026

## Project Overview
This repository contains our solution for the ALTEGRAD Data Challenge. The goal is to generate chemical textual descriptions for molecules given their graph representation ($G = (V, E)$). 

We explored two distinct paradigms to solve this problem:
1.  **Enhanced Retrieval:** Learning a metric space to retrieve the best caption from the training set.
2.  **Generative AI:** Building an Encoder-Decoder architecture (Graph-to-Sequence) to synthesize descriptions *ab initio*.

## Repository Structure

The project is organized into three main experimental folders. Below is the global file tree:

```text
.
├── upgraded_baseline/          # 🏆 Best Retrieval Approach (Score: 0.601)
│   └── train_gcn.py # Main inference script for submission
│
├── T5/                         # 🥈 Best Generative Approach (Score: 0.544)
│   ├── data/
│   ├── Output/
│   └── src/
│       ├── train_graph_t5.py   # Main training loop (Gradient Accum + FP16)
│       ├── graph_t5.py         # GraphT5Model Architecture (GINE + FLAN-T5)
│       ├── encoder.py          # Node-level GINE Encoder (No pooling)
│       └── data_loader.py      # Dataset class
│
├── Generative_approach/        # Early Experiments (BioGPT, Soft Prompting)
│   ├── data/
│   ├── Output/
│   └── src/
│       ├── train.py            # Training loop with Hybrid Alignment Loss
│       ├── model.py            # BioGPT + LoRA + Global Pooling Model
│       ├── generate_description_embeddings.py # Pre-computes BERT targets
│       └── encoder.py          # Standard GINE Encoder
│
└── MolTextNet/                 # External Data Experiments (Not used in final)
    ├── data/
    └── src/
        ├── preprocess_data.py  # Cleaning MolTextNet data
        └── train_model.py      # Pre-training scripts