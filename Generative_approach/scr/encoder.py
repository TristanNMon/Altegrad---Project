import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

# Feature Dimensions from Challenge PDF [cite: 46-55, 61-65]
NODE_FEAT_DIMS = [119, 9, 11, 12, 9, 5, 8, 2, 2]
EDGE_FEAT_DIMS = [22, 6, 2]

class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim) for dim in NODE_FEAT_DIMS
        ])

    def forward(self, x):
        out = 0
        for i, emb in enumerate(self.embeddings):
            # FIXED: Data is already indices. Do not add offset.
            val = x[:, i].long()
            
            # Safety: Clamp to ensure we don't crash on bad indices
            max_idx = self.embeddings[i].num_embeddings - 1
            val = torch.clamp(val, min=0, max=max_idx)
            
            out += emb(val)
        return out

class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim) for dim in EDGE_FEAT_DIMS
        ])

    def forward(self, edge_attr):
        out = 0
        for i, emb in enumerate(self.embeddings):
            val = edge_attr[:, i].long()
            max_idx = self.embeddings[i].num_embeddings - 1
            val = torch.clamp(val, min=0, max=max_idx)
            out += emb(val)
        return out

class GINEEncoder(nn.Module):
    """
    Graph Isomorphism Network with Edge Features.
    """
    def __init__(self, hidden_dim=256, num_layers=4, drop_ratio=0.1):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, batch):
        x = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, batch.edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_in # Residual connection

        return global_add_pool(x, batch.batch)