import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

from data_loader import GenerativeGraphDataset, collate_fn
from model import Graph2TextModel

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-4 # Slightly lower for pre-trained models
WEIGHT_DECAY = 1e-2
MODEL = 'biogpt'

PATHS = {
    "train": "../data/train_graphs.pkl",
    "val": "../data/validation_graphs.pkl",
    "test": "../data/test_graphs.pkl",
<<<<<<< HEAD
    "train_emb": f"../data/train_{MODEL}_embeddings.csv",
    "val_emb": f"../data/validation_{MODEL}_embeddings.csv",
    "save_path": f"checkpoints/best_model_{MODEL}.pt",
    "submission": f"../Output/submission_{MODEL}.csv"
>>>>>>>>> Temporary merge branch 2
}

def train_epoch(model, loader, optimizer, mse_criterion, epoch):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    # UNPACK 4 ITEMS
    for batch_graph, input_ids, attention_mask, target_bert_emb in loop:
        batch_graph = batch_graph.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        target_bert_emb = target_bert_emb.to(DEVICE)
        
        # FORWARD: Get outputs AND projected vector
        outputs, projected_emb = model(batch_graph, input_ids, attention_mask)
        
        # 1. Generation Loss (Write good English)
        gen_loss = outputs.loss 
        
        # 2. Alignment Loss (Match graph meaning to text meaning)
        align_loss = mse_criterion(projected_emb, target_bert_emb)
        
        # HYBRID LOSS
        loss = gen_loss + (10.0 * align_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), gen=gen_loss.item(), align=align_loss.item())
        
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, mse_criterion, epoch):
    model.eval()
    total_loss = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch_graph, input_ids, attention_mask, target_bert_emb in loop:
        batch_graph = batch_graph.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        target_bert_emb = target_bert_emb.to(DEVICE)
        
        outputs, projected_emb = model(batch_graph, input_ids, attention_mask)
        gen_loss = outputs.loss
        align_loss = mse_criterion(projected_emb, target_bert_emb)
        
        loss = gen_loss + (10.0 * align_loss)
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

@torch.no_grad()
def generate_submission(model, test_path, tokenizer, output_file):
    print(f"\nGenerative Inference on {test_path}...")
    model.eval()
    test_ds = GenerativeGraphDataset(test_path, split="test")
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn, shuffle=False)
    
    results = []
    for batch_graph, _, _, _ in tqdm(loader):
        batch_graph = batch_graph.to(DEVICE)
        output_tokens = model.generate_caption(batch_graph, tokenizer)
        caption = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        results.append({"ID": batch_graph.id[0], "description": caption})
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")

def main():
    # os.makedirs("checkpoints", exist_ok=True)
    
    print("Initializing Data Loaders with Embeddings...")
    train_ds = GenerativeGraphDataset(PATHS["train"], emb_path=PATHS["train_emb"], split="train")
    val_ds = GenerativeGraphDataset(PATHS["val"], emb_path=PATHS["val_emb"], split="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # print("Initializing LoRA Model...")
    model = Graph2TextModel().to(DEVICE)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # mse_criterion = nn.MSELoss()
    
    # best_val_loss = float("inf")
    # for epoch in range(1, EPOCHS + 1):
    #     train_loss = train_epoch(model, train_loader, optimizer, mse_criterion, epoch)
    #     val_loss = eval_epoch(model, val_loader, mse_criterion, epoch)
    #     print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), PATHS["save_path"])
            
    print("Generating Submission...")
    model.load_state_dict(torch.load(PATHS["save_path"]))
    generate_submission(model, PATHS["test"], train_ds.tokenizer, PATHS["submission"])

if __name__ == "__main__":
    main()