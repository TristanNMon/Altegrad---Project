import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

# =========================================================
# Feature Dimensions (Derived from data_utils.py maps)
# =========================================================
# Node feature sizes: [atomic_num, chirality, degree, charge, num_hs, radical, hybridization, aromatic, ring]
NODE_FEAT_DIMS = [119, 9, 11, 12, 9, 5, 8, 2, 2]

# Edge feature sizes: [bond_type, stereo, conjugated]
EDGE_FEAT_DIMS = [22, 6, 2]

class AtomEncoder(nn.Module):
    """
    Encodes the 9 categorical node features into a single vector.
    Reference: Section 2.2 of Challenge PDF.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim) for dim in NODE_FEAT_DIMS
        ])

    def forward(self, x):
        # x shape: [num_nodes, 9]
        out = 0
        for i, emb in enumerate(self.embeddings):
            # FIXED: Removed the '+ 5' logic for formal charge.
            # The data is already indices (0 to N-1), not raw values.
            val = x[:, i].long() 
            
            # Safety Clamp: Prevents crashing if data contains unexpected indices
            # This handles cases where 'atomic_num' might be 119 in a 119-size embedding
            if i < len(self.embeddings):
                 num_emb = self.embeddings[i].num_embeddings
                 val = torch.clamp(val, min=0, max=num_emb - 1)
            
            out += emb(val)
        return out


class BondEncoder(nn.Module):
    """
    Encodes the 3 categorical edge features into a single vector.
    Reference: Section 2.3 of Challenge PDF.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim) for dim in EDGE_FEAT_DIMS
        ])

    def forward(self, edge_attr):
        # edge_attr shape: [num_edges, 3]
        out = 0
        for i, emb in enumerate(self.embeddings):
            out += emb(edge_attr[:, i])
        return out


class GINEEncoder(nn.Module):
    """
    Graph Isomorphism Network with Edge Features.
    Translates the molecule graph into a single vector representation.
    """
    def __init__(self, hidden_dim=256, num_layers=4, drop_ratio=0.1):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # GINEConv requires an MLP (Multi-Layer Perceptron) to aggregate neighbors
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True)) # train_eps=True allows learning the epsilon scalar
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, batch):
        # 1. Encode Atoms and Bonds
        x = self.atom_encoder(batch.x)              # [num_nodes, hidden_dim]
        edge_attr = self.bond_encoder(batch.edge_attr) # [num_edges, hidden_dim]

        # 2. Message Passing
        for conv, norm in zip(self.convs, self.norms):
            # GINEConv expects edge_attr to be added to neighbor features
            x_in = x
            x = conv(x, batch.edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_in # Residual connection

        # 3. Graph Pooling (Readout)
        # Combines all atom vectors into one molecule vector
        graph_emb = global_add_pool(x, batch.batch) # [batch_size, hidden_dim]
        
        return graph_emb

# =========================================================
# Quick Test Block
# =========================================================
if __name__ == "__main__":
    # Create dummy data to test dimensions
    print("--- Encoder Test ---")
    emb_dim = 64
    model = GINEEncoder(hidden_dim=emb_dim)
    
    # Dummy Batch: 2 molecules
    # Mol 1: 2 atoms, 1 bond
    # Mol 2: 1 atom, 0 bonds
    from torch_geometric.data import Data, Batch
    
    # Random features roughly within range
    d1 = Data(x=torch.randint(0, 2, (2, 9)), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randint(0, 2, (2, 3)))
    d2 = Data(x=torch.randint(0, 2, (1, 9)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 3), dtype=torch.long))
    
    batch = Batch.from_data_list([d1, d2])
    
    output = model(batch)
    print(f"Output Shape: {output.shape}") # Should be [2, 64]
    print("Success!")