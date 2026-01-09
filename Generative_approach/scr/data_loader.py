import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

class GenerativeGraphDataset(Dataset):
    """
    Dataset for Graph-to-Text Generation.
    
    Returns:
        - graph: PyTorch Geometric Data object
        - input_ids: Token indices for the caption (for the LLM)
        - attention_mask: Mask to ignore padding tokens
    """
    def __init__(self, graph_path, tokenizer_name="distilgpt2", max_len=128, split="train"):
        print(f"Loading {split} graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        # Load the tokenizer (using DistilGPT2 by default as it's efficient)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # GPT-2 does not have a pad token by default, so we use the EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_len = max_len
        self.split = split
        print(f"Loaded {len(self.graphs)} graphs for {split}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        
        # Prepare the text target
        if self.split == "test":
            # Test set has no descriptions, return dummy text or just the graph
            text = ""
        else:
            text = graph.description

        # Tokenize the text
        # We add the EOS token to the end so the model learns when to stop generating
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return graph, input_ids, attention_mask

def collate_fn(batch):
    """
    Custom collate function to handle variable size graphs and token sequences.
    """
    graphs, input_ids, attention_masks = zip(*batch)
    
    # Batch the graphs into a single large graph (standard PyG practice)
    batch_graph = Batch.from_data_list(list(graphs))
    
    # Stack the text tensors (they are already padded to max_len)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    
    return batch_graph, input_ids, attention_masks

# =========================================================
# Quick Test Block
# =========================================================
if __name__ == "__main__":
    # Point this to your actual data path to test
    dummy_path = "../data/train_graphs.pkl" 
    
    try:
        ds = GenerativeGraphDataset(dummy_path, max_len=64)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        
        batch_graph, batch_ids, batch_mask = next(iter(loader))
        
        print("\n--- Data Loader Test ---")
        print(f"Batch Graphs: {batch_graph}") 
        print(f"Batch IDs Shape: {batch_ids.shape}") # Should be [4, 64]
        print(f"Decoded first sample: {ds.tokenizer.decode(batch_ids[0], skip_special_tokens=True)}")
        print("Success!")
    except FileNotFoundError:
        print(f"Could not find {dummy_path}, skipping test.")