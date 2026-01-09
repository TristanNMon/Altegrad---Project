import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

# Import our custom modules
from data_loader import GenerativeGraphDataset, collate_fn
from model import Graph2TextModel

# =========================================================
# Configuration
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-4 # Slightly lower for pre-trained models
WEIGHT_DECAY = 1e-2

PATHS = {
    "train": "../data/train_graphs.pkl",
    "val": "../data/validation_graphs.pkl",
    "test": "../data/test_graphs.pkl",
    "save_path": "checkpoints/best_model.pt",
    "submission": "submission.csv"
}

# =========================================================
# Training Loop
# =========================================================
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_graph, input_ids, attention_mask in loop:
        # Move data to GPU
        batch_graph = batch_graph.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        # Forward pass (Auto-calculates Loss because we provide labels)
        outputs = model(batch_graph, input_ids, attention_mask)
        loss = outputs.loss # CrossEntropyLoss provided by Hugging Face
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

# =========================================================
# Validation Loop
# =========================================================
@torch.no_grad()
def eval_epoch(model, loader, epoch):
    model.eval()
    total_loss = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch_graph, input_ids, attention_mask in loop:
        batch_graph = batch_graph.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        outputs = model(batch_graph, input_ids, attention_mask)
        loss = outputs.loss
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

# =========================================================
# Inference / Generation (Kaggle Submission)
# =========================================================
@torch.no_grad()
def generate_submission(model, test_path, tokenizer, output_file):
    print(f"\nGenerative Inference on {test_path}...")
    model.eval()
    
    # Load Test Data (No shuffle, batch_size=1 for safety in generation)
    test_ds = GenerativeGraphDataset(test_path, split="test")
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn, shuffle=False)
    
    results = []
    
    for batch_graph, _, _ in tqdm(loader, desc="Generating Captions"):
        batch_graph = batch_graph.to(DEVICE)
        
        # Beam Search Generation
        output_tokens = model.generate_caption(
            batch_graph, 
            tokenizer, 
            max_length=128, 
            num_beams=5 # Higher beam = better quality, slower
        )
        
        # Decode tokens to text
        # skip_special_tokens removes <s>, </s>, <pad>
        caption = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Store result with ID
        # Note: batch.id is a list of IDs in the batch
        results.append({
            "ID": batch_graph.id[0], 
            "description": caption
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")

# =========================================================
# Main Function
# =========================================================
def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Load Data
    print("Initializing Data Loaders...")
    train_ds = GenerativeGraphDataset(PATHS["train"], split="train")
    val_ds = GenerativeGraphDataset(PATHS["val"], split="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 2. Initialize Model
    print("Initializing Model...")
    model = Graph2TextModel().to(DEVICE)
    
    # AdamW is standard for Transformers (handles weight decay better)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 3. Training Loop
    best_val_loss = float("inf")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        val_loss = eval_epoch(model, val_loader, epoch)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PATHS["save_path"])
            print(">>> New Best Model Saved!")
            
    # 4. Generate Submission
    print("\nTraining Complete. Loading Best Model for Generation...")
    model.load_state_dict(torch.load(PATHS["save_path"]))
    
    generate_submission(
        model, 
        PATHS["test"], 
        train_ds.tokenizer, 
        PATHS["submission"]
    )

if __name__ == "__main__":
    main()