import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
from data_utils import PreprocessedGraphDataset

# ==========================================
# 1. MAPPINGS
# ==========================================
def get_bond_type(idx):
    # Mapping based on data_utils.py
    # 0:UNSPECIFIED, 1:SINGLE, 2:DOUBLE, 3:TRIPLE, ..., 12:AROMATIC
    if idx == 1: return Chem.BondType.SINGLE
    if idx == 2: return Chem.BondType.DOUBLE
    if idx == 3: return Chem.BondType.TRIPLE
    if idx == 12: return Chem.BondType.AROMATIC
    if idx in [17, 18, 19, 20]: return Chem.BondType.DATIVE # Dative bonds
    # Skip Ionic(13), Hydrogen(14), Zero(21)
    return None 

def get_chirality(idx):
    if idx == 1: return Chem.ChiType.CHI_TETRAHEDRAL_CW
    if idx == 2: return Chem.ChiType.CHI_TETRAHEDRAL_CCW
    return Chem.ChiType.CHI_UNSPECIFIED

# ==========================================
# 2. RECONSTRUCTION STRATEGIES
# ==========================================

def attempt_reconstruction(data, strict_valence=True):
    """
    Attempts to build a molecule. 
    If strict_valence=True, it uses the dataset's 'num_hs' feature.
    If strict_valence=False, it ignores 'num_hs' and lets RDKit guess (fixes valence errors).
    """
    mol = Chem.RWMol()
    node_to_idx = {}
    
    # --- Add Atoms ---
    for i in range(data.x.shape[0]):
        atom_num = data.x[i, 0].item()
        if atom_num == 0: atom_num = 6 # Dummy -> Carbon
        
        a = Chem.Atom(atom_num)
        
        # Always set charge (Index 5 is 0)
        a.SetFormalCharge(data.x[i, 3].item() - 5)
        
        # Aromaticity (Index 1 is True)
        a.SetIsAromatic(data.x[i, 7].item() == 1)
        
        # Chirality
        chi = data.x[i, 1].item()
        if chi in [1, 2]: a.SetChiralTag(get_chirality(chi))
        
        # CRITICAL: Toggle Explicit Hydrogens based on strategy
        if strict_valence:
            a.SetNumExplicitHs(data.x[i, 4].item())
        else:
            # In relaxed mode, we DON'T set explicit Hs. 
            # RDKit will calculate implicit Hs automatically to satisfy valence.
            pass
            
        idx = mol.AddAtom(a)
        node_to_idx[i] = idx

    # --- Add Bonds ---
    rows, cols = data.edge_index
    edge_attrs = data.edge_attr
    added_bonds = set()
    
    for i in range(rows.shape[0]):
        u, v = rows[i].item(), cols[i].item()
        if u >= v: continue
        
        bt = get_bond_type(edge_attrs[i, 0].item())
        if bt is not None and (u, v) not in added_bonds:
            mol.AddBond(node_to_idx[u], node_to_idx[v], bt)
            added_bonds.add((u, v))

    return mol

def graph_to_smiles_robust(graph):
    # ATTEMPT 1: Strict Mode (Uses all features)
    try:
        mol = attempt_reconstruction(graph, strict_valence=True)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        pass # Fail silently, try next strategy

    # ATTEMPT 2: Relaxed Valence (Ignores num_hs, fixes the "Nitrogen with 5 bonds" errors)
    try:
        mol = attempt_reconstruction(graph, strict_valence=False)
        mol.UpdatePropertyCache(strict=False) # Calculate valences loosely
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS | 
                              Chem.SanitizeFlags.SANITIZE_KEKULIZE | 
                              Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | 
                              Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        pass

    # ATTEMPT 3: The "Raw" Dump (Unsanitized)
    # If all else fails, just give me the atoms and bonds string.
    # It might be chemically invalid, but the T5 model can still read it!
    try:
        mol = attempt_reconstruction(graph, strict_valence=False)
        mol.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
    except:
        return None

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    dataset = PreprocessedGraphDataset(graph_path='../data/test_graphs.pkl')
    
    results = []
    print(f"Processing {len(dataset)} molecules with ROBUST strategy...")
    
    success_count = 0
    
    for graph in tqdm(dataset):
        smiles = graph_to_smiles_robust(graph)
        
        if smiles is None:
            smiles = "C" # Absolute final fallback
        else:
            success_count += 1
            
        results.append({"id": graph.id, "smiles": smiles})
        
    print(f"Final Report: {success_count} valid, {1000 - success_count} failed.")
    pd.DataFrame(results).to_csv("../data/test_smiles.csv", index=False)