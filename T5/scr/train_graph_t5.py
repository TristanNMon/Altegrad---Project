import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

from data_loader import GenerativeGraphDataset, collate_fn
from graph_t5 import GraphT5Model

# =========================================================
# Configuration
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4 # T5 likes slightly smaller LR than pure GPT
WEIGHT_DECAY = 1e-4

PATHS = {
    "train": "../data/train_graphs.pkl",
    "val": "../data/validation_graphs.pkl",
    "test": "../data/test_graphs.pkl",
    "save_path": "checkpoints/best_graph_t5.pt",
    "submission": "../Output/submission_t5.csv"
}

# =========================================================
# Training Loop
# =========================================================
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    
    # We don't need BERT embeddings here, T5 handles the semantic loss internally
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_graph, input_ids, attention_mask, _ in loop:
        batch_graph = batch_graph.to(DEVICE)
        
        # T5 expects labels where padding is -100
        labels = input_ids.to(DEVICE)
        labels[labels == model.t5.config.pad_token_id] = -100
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_graph, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, epoch):
    model.eval()
    total_loss = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch_graph, input_ids, attention_mask, _ in loop:
        batch_graph = batch_graph.to(DEVICE)
        labels = input_ids.to(DEVICE)
        labels[labels == model.t5.config.pad_token_id] = -100
        
        outputs = model(batch_graph, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

@torch.no_grad()
def generate_submission(model, test_path, tokenizer, output_file):
    print(f"\nGenerative Inference on {test_path}...")
    model.eval()
    
    test_ds = GenerativeGraphDataset(test_path, split="test", tokenizer_name="t5-small")
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn, shuffle=False)
    
    results = []
    
    for batch_graph, _, _, _ in tqdm(loader):
        batch_graph = batch_graph.to(DEVICE)
        
        output_tokens = model.generate_caption(
            batch_graph, 
            tokenizer, 
            max_length=128, 
            num_beams=5
        )
        
        caption = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        results.append({"ID": batch_graph.id[0], "description": caption})
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")

# =========================================================
# Main
# =========================================================
def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Load Data
    # Important: Use 't5-small' tokenizer!
    print("Initializing Data Loaders...")
    train_ds = GenerativeGraphDataset(PATHS["train"], tokenizer_name="t5-small", split="train")
    val_ds = GenerativeGraphDataset(PATHS["val"], tokenizer_name="t5-small", split="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 2. Initialize Model
    print("Initializing Graph-T5 Model...")
    model = GraphT5Model(t5_model_name="t5-small").to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 3. Train
    best_val_loss = float("inf")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        val_loss = eval_epoch(model, val_loader, epoch)
        
        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PATHS["save_path"])
            print(">>> New Best Model Saved!")
            
    # 4. Generate
    print("Loading Best Model...")
    model.load_state_dict(torch.load(PATHS["save_path"]))
    generate_submission(model, PATHS["test"], train_ds.tokenizer, PATHS["submission"])

if __name__ == "__main__":
    main()