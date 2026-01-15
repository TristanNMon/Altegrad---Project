# Generative Molecular Captioning: Technical Report

## 1. Overview
We implemented an end-to-end **Generative Artificial Intelligence** pipeline to solve the captioning challenge. Unlike the retrieval baseline, this approach synthesizes descriptions *ab initio* using an Encoder-Decoder architecture, allowing for zero-shot generalization to novel chemical structures.

## 2. Module I: Data Preparation
**Objective:** Transition from embedding-based retrieval to token-based generation.

* **Tokenizer Integration:** We utilized the `distilgpt2` tokenizer via `AutoTokenizer`. This ensures exact compatibility with the pre-trained Language Model's vocabulary.
* **Padding Strategy:** Since causal LLMs lack default padding, we explicitly assigned the EOS token as the padding token to enable stable batch processing.
* **Graph Batching:** Implemented a custom `collate_fn` using `Batch.from_data_list`. This merges variable-sized graphs into a single super-graph, enabling efficient parallelization on GPUs without manual padding of node features.

## 3. Module II: The Graph Encoder ("The Chemist")
**Objective:** Extract chemically rich features using a structure-aware neural network.

* **Architecture (GINE):** We replaced the baseline GCN with a **Graph Isomorphism Network with Edge features (GINE)**. GINE is chemically superior as it explicitly weights message passing using bond attributes (e.g., aromatic vs. single bonds), preserving critical structural information that standard GCNs average out.
* **Feature Embedding:** We implemented `AtomEncoder` and `BondEncoder` to independently embed all 9 categorical node features and 3 edge features. These are summed to create a dense representation that distinguishes complex states (e.g., an aromatic Carbon vs. an aliphatic Carbon).
* **Deep Stacking:** Residual connections (`x = x + x_in`) allowed us to stack 4 GINE layers, capturing global substructures (like macrocycles) while mitigating the vanishing gradient problem.

## 4. Module III: The Decoder ("The Author")
**Objective:** Translate chemical embeddings into natural language via Transfer Learning.

* **Soft Prompting:** Instead of discrete translation, we project the global graph embedding into the LLM's dimension ($d=768$) and prepend it to the text sequence. This treats the molecule as a "visual token," allowing the LLM to attend to the structure directly.
* **Backbone:** We selected **DistilGPT-2** (6 layers) over GPT-2 (12 layers). This choice reduces computational overhead by ~50% while retaining sufficient pre-trained knowledge of English grammar to generate fluent descriptions.
* **Repetition Control:** A `repetition_penalty` of 1.2 was applied during generation to effectively suppress the "looping" failure mode common in captioning tasks.

## 5. Module IV: Training & Inference
**Objective:** Optimize the pipeline and generate high-quality captions.

* **Optimization:** We utilized **AdamW** instead of SGD. This decoupled weight decay is critical for fine-tuning Transformers, preventing catastrophic forgetting of the pre-trained linguistic manifold.
* **Teacher Forcing:** Training utilized teacher forcing (feeding ground-truth history) to stabilize convergence and speed up learning.
* **Beam Search:** For inference, we replaced greedy decoding with **Beam Search** ($k=5$). This explores multiple probability paths simultaneously, ensuring the final output is globally coherent rather than just locally probable.