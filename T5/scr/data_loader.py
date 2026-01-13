import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from transformers import AutoTokenizer

class GenerativeGraphDataset(Dataset):
    """
    Dataset that loads graphs, tokenizes text, and retrieves pre-computed BioGPT embeddings.
    """
    def __init__(self, graph_path, emb_path=None, tokenizer_name="microsoft/biogpt", max_len=256, split="train"):
        print(f"Loading {split} graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        # Load Embeddings (BioGPT vectors are 1024-dim)
        self.id2emb = {}
        if emb_path:
            print(f"Loading embeddings from {emb_path}...")
            df = pd.read_csv(emb_path)
            for _, row in df.iterrows():
                emb_vals = np.fromstring(row["embedding"], sep=',')
                self.id2emb[str(row["ID"])] = torch.tensor(emb_vals, dtype=torch.float32)
        
        # Initialize Tokenizer (BioGPT)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_len = max_len
        self.split = split

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        
        # 1. Prepare Text
        if self.split == "test":
            text = ""
        else:
            text = graph.description

        # 2. Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 3. Retrieve Embedding (Alignment Target)
        if hasattr(self, 'id2emb') and str(graph.id) in self.id2emb:
            bert_emb = self.id2emb[str(graph.id)]
        else:
            # FIX: BioGPT embeddings are 1024, not 768.
            # If you leave this as 768, MSELoss will crash on dimension mismatch.
            bert_emb = torch.zeros(1024) 
        
        return graph, input_ids, attention_mask, bert_emb

def collate_fn(batch):
    graphs, input_ids, att_masks, bert_embs = zip(*batch)
    
    batch_graph = Batch.from_data_list(list(graphs))
    input_ids = torch.stack(input_ids)
    att_masks = torch.stack(att_masks)
    bert_embs = torch.stack(bert_embs)
    
    return batch_graph, input_ids, att_masks, bert_embs