# MolTextNet Data Experiments

This folder contains experimental code for processing the MolTextNet dataset (2.5M molecule-description pairs).

## Context
We explored the possibility of pre-training our Graph Encoder on a massive external dataset to learn robust chemical features before fine-tuning on the challenge data.

* `preprocess_data.py`: Scripts to clean and tokenize SMILES/Text pairs from MolTextNet.
* `train_model.py`: Pre-training loop.

**Note:** This approach was investigated but **not included in the final submission** to strictly adhere to the "No External Data" rule of the ALTEGRAD challenge.