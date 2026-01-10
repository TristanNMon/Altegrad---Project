#!/usr/bin/env python3
"""Generate BioGPT embeddings for molecular descriptions."""

import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# Configuration
MAX_TOKEN_LENGTH = 256 
MODEL_NAME = "microsoft/biogpt" # <--- CHANGED TO BIOGPT (1024 dim)

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = model.to(device)
model.eval()
print(f"Model loaded on: {device}")

# Process each split
for split in ['train', 'validation']:
    print(f"\nProcessing {split}...")
    
    # Path adjustment logic
    pkl_path = f'../data/{split}_graphs.pkl'
    if not os.path.exists(pkl_path):
        pkl_path = f'data/{split}_graphs.pkl'
        
    print(f"Loading from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        graphs = pickle.load(f)
    
    ids = []
    embeddings = []
    
    for graph in tqdm(graphs, total=len(graphs)):
        description = graph.description
        
        # Tokenize
        inputs = tokenizer(
            description, 
            return_tensors='pt', 
            truncation=True, 
            max_length=MAX_TOKEN_LENGTH, 
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # MEAN POOLING: Average all tokens to get a 1024-dim vector
        # This is better than [0] for causal models like BioGPT
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        # Sum of token embeddings
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1)
        # Count of real tokens
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        
        # Divide to get mean
        embedding = (sum_embeddings / sum_mask).cpu().numpy().flatten()
        
        ids.append(graph.id)
        embeddings.append(embedding)
    
    result = pd.DataFrame({
        'ID': ids,
        'embedding': [','.join(map(str, emb)) for emb in embeddings]
    })
    
    output_path = f'../data/{split}_biogpt_embeddings.csv'
    result.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

print("\nDone!")