# Phase III: Graph-to-Sequence T5 (Best Generative Model)

This folder contains our most advanced generative architecture, which treats molecular captioning as a translation task.

## Method
* **Architecture:** Node-Level Graph-to-Sequence.
* **Encoder:** GINE (No Pooling). Outputs a sequence of $N$ atom vectors.
* **Decoder:** Google FLAN-T5 (Encoder-Decoder Transformer).
* **Mechanism:** Cross-Attention. The T5 decoder attends to specific atoms in the sequence to generate corresponding text features.
* **Optimization:** Trained with **LoRA** (Low-Rank Adaptation) and **Mixed Precision (FP16)**. Inference uses **Beam Search** ($k=10$).

## Key Files
* `train_graph_t5.py`: Memory-optimized training loop (Gradient Accumulation + FP16).
* `graph_t5.py`: The `GraphT5Model` class that fuses the GINE encoder with the T5 backbone.
* `encoder.py`: The specific GINE implementation returning node sequences (no global pooling).

## Results
* **Score:** **0.544** (Best Generative Result)
* **Insight:** Outperformed BioGPT by ~20% by removing the information bottleneck.