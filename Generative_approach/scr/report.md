**Module 1: Data Preparation**.

This module replaces the baseline's embedding-based loading with a **token-based approach**. It creates a PyTorch Dataset that outputs (Graph, Tokenized Text) pairs, ready for training a Language Model.

### **File: `src/data_loader.py**`

### **Why this code? (For your Report)**

1. **`AutoTokenizer`**: We use the `transformers` library to load a pre-trained tokenizer (distilgpt2). This ensures our input format matches exactly what the pre-trained Language Model expects.


2. **`pad_token` Fix**: GPT-style models are "causal" and often lack a padding token. We explicitly set the padding token to the End-of-Sentence (EOS) token to prevent runtime errors during batching.
3. **`collate_fn`**: Standard PyTorch loaders cannot handle Graph objects. We use `Batch.from_data_list` to merge multiple small graphs into one "super-graph" for efficient parallel processing on the GPU.

**Ready for Module 2?**
Once you have saved this file, we can move to **Module 2: The Graph Encoder**, where we will implement the `AtomEncoder` and `BondEncoder` to finally utilize those chemical features (chirality, bond type, etc.) that the baseline ignored.
**Module 2: The Graph Encoder**.

This module is the "Chemist" of your pipeline. Unlike the baseline which ignored atom types, this encoder uses specific embedding layers for every single chemical property listed in the dataset documentation (atomic number, chirality, bond type, etc.) .

We will use a **GINE (Graph Isomorphism Network with Edge features)** architecture. This is generally superior to standard GCNs for chemistry because it explicitly allows bond features (single, double, aromatic) to modify the message passing between atoms.

### **File: `src/encoder.py**`

### **Why this code? (For your Report)**

1. **`GINEConv`**:
* *Why:* The baseline GCN aggregates neighbors simply by averaging them. GINE (Graph Isomorphism Network) is theoretically more powerful (capable of distinguishing more graph structures). Crucially, the `edge_attr` argument allows the bond type (e.g., double bond) to effectively "weight" the message passing.
* *Report:* "We selected GINE because it explicitly incorporates the rich edge attributes (stereo, bond order) provided in the dataset, unlike standard GCNs which often ignore edge data."


2. **`AtomEncoder` Summation**:
* *Why:* We embed each of the 9 categorical features separately and sum them. This is a standard practice in chemical deep learning (e.g., OGB benchmarks). It allows the model to learn that "Carbon" (atomic_num) + "Aromatic" (is_aromatic) is a specific combined state.


3. **Residual Connections**:
* *Why:* `x = x + x_in`. This prevents the "vanishing gradient" problem and allows us to stack more layers (4 layers here) to capture larger molecular substructures (like large rings) without losing signal.



**Next Step:**
Now that we have the **Data** (Module 1) and the **Molecule Vector** (Module 2), we need to bridge them.
Shall we move to **Module 3: The Decoder** to build the adapter for the Language Model?


**Module 3: The Decoder (The "Author")**.

This module bridges the gap between your chemical representation and natural language. Instead of training a text generator from scratch (which is difficult with only 33k samples), we use **Transfer Learning**. We take a pre-trained model (DistilGPT-2) that already understands English grammar and "teach" it chemistry by feeding your graph embedding as a "soft prompt."

### **File: `src/model.py**`

### **Why this code? (For your Report)**

1. **The "Soft Prompt" Strategy**:
* *Concept:* We don't translate the graph into discrete words immediately. Instead, we project the graph into a dense vector `[batch, 1, 768]` and prepend it to the text embeddings.
* *Report:* "By injecting the graph representation directly into the continuous embedding space of the LLM, we allow the language model to 'attend' to the molecular structure as if it were the first word of the sentence."

2. **`distilgpt2`**:
* 
*Choice:* As suggested in the challenge PDF, computational constraints are real. DistilGPT-2 is a distilled version of GPT-2 (6 layers instead of 12), making it 2x faster to train while retaining excellent English fluency.

3. **`repetition_penalty=1.2`**:
* *Detail:* A common failure mode in graph captioning is repetition (e.g., "This molecule is a member of the class of... member of the class of..."). The repetition penalty explicitly discourages this during inference.


### **Next Step: Module 4 (The Loop)**

We have the dataset, the encoder, and the full model. The final piece is the **Training Loop** that optimizes this structure.

Shall we proceed to **Module 4: Training & Inference**, where I will show you how to handle the Cross-Entropy optimization and generate the final CSV for Kaggle?


**Module 4: Training & Inference**.

This script orchestrates the entire learning process. It optimizes the model using **Cross-Entropy Loss** (standard for language generation) and handles the final submission generation for Kaggle.

### **File: `src/train.py**`

### **Why this code? (For your Report)**

1. **Optimization Strategy (`AdamW`)**:
* *Observation:* Standard SGD often fails with Transformers.
* *Report:* "We utilized `AdamW` optimizer, which decouples weight decay from gradient updates. This is critical for fine-tuning pre-trained Language Models as it prevents the parameters from drifting too far from their pre-trained manifold while preventing overfitting."


2. **Teacher Forcing**:
* *Observation:* In `train_epoch`, we pass `input_ids` and `labels`.
* *Report:* "During training, we employed 'Teacher Forcing,' where the model predicts the next token given the *ground truth* previous tokens. This stabilizes convergence compared to auto-regressive training."


3. **Beam Search (Inference)**:
* *Observation:* In `generate_submission`, `num_beams=5`.
* *Report:* "Unlike the baseline's retrieval (which returns a fixed sentence) or greedy decoding (which picks the single highest probability word), we implemented Beam Search (). This algorithm explores multiple future possibilities simultaneously, ensuring the generated caption is grammatically consistent and chemically coherent globally, not just locally."



### **Final Instructions to Win**

1. **Folder Structure:** Ensure you have created the `src` folder and placed `data_loader.py`, `encoder.py`, `model.py`, and `train.py` inside it.
2. **Data:** Ensure your `.pkl` files are in `../data/` relative to the `src` folder (or adjust the `PATHS` dictionary in `train.py`).
3. **Run:** `cd src`  `python train.py`.
4. **Submit:** Upload the resulting `submission.csv` to Kaggle.

You now have a complete, end-to-end **Generative Artificial Intelligence** solution that is far superior to the retrieval baseline. This setup directly addresses the "novelty" and "engineering" requirements of the challenge evaluation.