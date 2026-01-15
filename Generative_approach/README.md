# Phase I & II: Generative Experiments (BioGPT & Alignment)

This folder contains our initial attempts at generative captioning using **Global Pooling** and **Causal LLMs**.

## Method
* **Encoder:** GINE with Global Mean Pooling ($R^{Graph} \to R^{256}$).
* **Decoder:** BioGPT or DistilGPT-2 (Causal Language Models).
* **Technique:** Soft Prompting (Prefix Tuning).
* **Optimization:** We implemented an auxiliary **Teacher-Student Alignment Loss** where the graph encoder is forced to mimic **ModernBERT** text embeddings to reduce hallucinations.

## Key Files
* `train.py`: Main training loop implementing the Hybrid Loss (Generation + Alignment).
* `model.py`: Architecture definition (GINE + Projector + LoRA-tuned LLM).
* `generate_description_embeddings.py`: Script to pre-compute ModernBERT embeddings for the alignment loss.
* `data_loader.py`: Custom PyTorch dataset handling graph loading and BERT targets.

## Results
* **Best Score:** 0.488 (DistilGPT-2)
* **Limitation:** The Global Pooling bottleneck caused the model to lose count of atoms, leading to hallucinations. This motivated the shift to the T5 approach.