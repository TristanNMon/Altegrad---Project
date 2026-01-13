Based on the challenge description file, here is a strategic roadmap of how I would approach the ALTEGRAD Molecular Graph Captioning challenge.

The core of this challenge is bridging two modalities: **Graph Representation Learning** (molecules) and **Natural Language Processing** (text captions).

### Phase 1: Analyze & Exploit the Data

Before building complex models, I would maximize the value extracted from the provided data structures. The dataset is relatively small (~33k pairs), meaning feature engineering and efficient architectures are crucial.

* **Rich Feature Utilization:** The dataset provides specific chemical features (chirality, hybridization, bond types, etc.). I would ensure my model uses specific embedding layers for each of the 9 node features and 3 edge features rather than summing them or treating them generically.


* **SMILES Conversion (Data Augmentation):** Since you have the graph structure (atoms and bonds), I would write a script to convert these graphs back into **SMILES strings** (linear text representations of molecules) using RDKit.
* *Why?* This turns the problem into a "Translation" task (SMILES string  Description text), allowing you to leverage powerful sequence-to-sequence Transformers (like T5 or BART) alongside your Graph Neural Network (GNN).



### Phase 2: Move Beyond the Baseline

The provided baseline is a **retrieval model** that finds the "nearest neighbor" caption from the training set. This is limited because it cannot generate new descriptions for novel molecular structures.

To win, you must follow the suggestion to build a **Generative Model**. Here is the architecture I would propose:

#### 1. The Encoder (The "Eye")

Replace the simple GCN from the baseline  with a more chemically-aware GNN.

* **Recommendation:** Use **GINE (Graph Isomorphism Network with Edge features)** or a **Graph Transformer**.
* **Reasoning:** The dataset has critical edge features (bond type, stereochemistry). Standard GCNs often struggle to fully utilize edge attributes. A Graph Transformer can also capture long-range dependencies between atoms that are far apart in the graph but close in 3D space.



#### 2. The Decoder (The "Voice")

You need a model that takes the graph embeddings and outputs a sequence of text.

* **Option A (Lightweight):** An LSTM or GRU with an **Attention Mechanism**. The attention is vital—it allows the model to "focus" on specific parts of the molecule (e.g., a specific ring or functional group) when generating the corresponding word (e.g., "aromatic", "hydroxyl").
* **Option B (High Performance):** A Transformer Decoder (like GPT-2). You project the graph embeddings into the same dimension as the word embeddings and feed them as "soft prompts" or prefixes to the Transformer.

### Phase 3: Advanced Strategies (The "Winning" Edge)

If I were competing, I would implement **Cross-Modal Alignment** or a **RAG-style approach**:

**1. Soft-Prompting Pre-trained LLMs**
The PDF allows external pre-trained models.

* **Strategy:** Take a pre-trained LLM (like GPT-2 Medium as suggested  or DistilGPT-2).


* **Connection:** Train a lightweight "Adapter" network (a simple MLP) that takes your GNN's output and projects it into the LLM's input space.
* **Benefit:** The LLM already knows how to write fluent English. You only need to teach it to "read" the chemistry from your adapter.

**2. Joint Training (Contrastive + Generative)**
Don't throw away the baseline's logic!

* **Strategy:** Train your model with a multi-task loss.
1. **Generation Loss:** Standard Cross-Entropy (predict the next word).
2. **Contrastive Loss:** Ensure the *global* embedding of the generated sentence is close to the *global* embedding of the graph (similar to the baseline's MSE approach).




* **Benefit:** This keeps the text semantically aligned with the molecule while ensuring grammatical correctness.

**3. Hard-Retrieval Augmented Generation**

* **Strategy:** Use the baseline retrieval model to find the most similar molecule in the training set and its caption.
* **Input to Model:** `[Graph Embedding] + [Retrieved Caption]`
* **Output:** `[Final Caption]`
* **Logic:** It is easier for the model to "edit" an existing, correct caption (e.g., changing "methyl" to "ethyl") than to write one from scratch.

### Summary Checklist for Success

1. [ ] **Run Baseline:** Confirm the pipeline works and check the Leaderboard evaluation script.
2. [ ] **Upgrade Encoder:** Swap GCN for GATv2 or Graph Transformer to better handle the edge attributes (bond types/stereo).


3. [ ] **Build Generator:** Implement an Encoder-Decoder architecture with Attention.
4. [ ] **Evaluate Locally:** Monitor **BLEU-4** (text overlap) and **BERTScore** (semantic meaning) as these are the official metrics.


5. [ ] **Report:** Document "what didn't work" (e.g., if simple GCNs failed to capture chirality) as explicitly requested for the report.


# ALTEGRAD 2025: Molecular Graph Captioning (Generative Approach)

## 🧪 Project Overview
This project addresses the ALTEGRAD Data Challenge: **Molecular Graph Captioning**. The goal is to "translate" a molecule's 2D graph structure (atoms and bonds) into a coherent, human-readable text description.

While the provided baseline treats this as a **Retrieval Task** (finding the closest existing caption), this repository implements a **Generative Model** (Encoder-Decoder) to synthesize new descriptions from scratch. This approach allows for the description of novel molecular structures not seen during training.

---

## 🏗️ Architecture & Philosophy

Our approach bridges Graph Representation Learning and Natural Language Processing (NLP) through a modular pipeline. We deviate from the baseline by treating the problem as a sequence-to-sequence task rather than an embedding-similarity task.

### Module 1: Data Preparation (The "Translator")
* **Objective:** Convert raw graph data and text into trainable sequences.
* **The Shift:**
    * *Baseline:* Used pre-computed BERT embeddings (fixed vectors).
    * *Our Approach:* Implements a **Tokenizer** (e.g., GPT-2 Tokenizer). We treat text as a sequence of tokens to train the model on next-word prediction.
* **Report Justification:** "Generative modeling requires token-level supervision to learn syntax and chemical terminology, whereas static embeddings lose fine-grained lexical details."

### Module 2: The Graph Encoder (The "Chemist")
* **Objective:** Create a chemically accurate vector representation of the molecule.
* **The Shift:**
    * *Baseline:* Vanilla GCN that ignored atom types (random initialization) and edge features.
    * *Our Approach:* **GINE (Graph Isomorphism Network with Edge Features)** or **Graph Transformer**. We strictly map the specific features provided (Chirality, Hybridization, Bond Type) to learnable embeddings.
* **Report Justification:** "Standard GCNs fail to capture stereochemistry (cis/trans isomers) and bond types (single vs. double), which are critical for describing molecular geometry and reactivity."

### Module 3: The Decoder (The "Author")
* **Objective:** Translate the chemical representation into fluent English.
* **The Shift:**
    * *Baseline:* None (Retrieval only).
    * *Our Approach:* **Soft-Prompting a Pre-trained LLM (DistilGPT-2)**. The graph embedding is projected into the LLM's input space and acts as a "prefix" or "context" token.
* **Report Justification:** "Training a decoder from scratch on ~33k samples risks overfitting. Leveraging a pre-trained Language Model (Transfer Learning) provides a strong prior for English grammar, allowing the model to focus on learning chemical content."

### Module 4: Training & Inference (The "Loop")
* **Objective:** Optimize the generation quality.
* **The Shift:**
    * *Baseline:* MSE Loss (distance in latent space).
    * *Our Approach:* **Cross-Entropy Loss** (maximizing likelihood of the correct next word) + **Beam Search** for inference.
* **Report Justification:** "MSE is a poor proxy for text quality. Cross-Entropy directly aligns the model with the discrete nature of language generation."

---

## 📂 Project Structure

```text
├── data/
│   ├── train_graphs.pkl      # PyTorch Geometric Data objects
│   ├── test_graphs.pkl       # (Description hidden)
│   └── ...
├── src/
│   ├── data_loader.py        # GenerativeGraphDataset & Tokenizer logic
│   ├── encoder.py            # AtomEncoder, BondEncoder, GINE/GAT architecture
│   ├── decoder.py            # Adapter layer & Pre-trained LLM wrapper
│   └── train.py              # Training loop with Cross-Entropy Loss
├── generate.py               # Inference script using Beam Search
└── README.md                 # Project documentation

