import os

import torch
import torch_geometric
import numpy as np

import pandas as pd
import pickle

from copy import deepcopy
import rdkit

import os
import pickle

class ChIRo(torch_geometric.data.Dataset):
    @classmethod
    def load_data(cls, path='data/raw_data/chiro/train.pkl', regression=''):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        df = obj
        print(f"Loaded DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")


        # --- Validate mol column exists ---
        if 'rdkit_mol_cistrans_stereo' not in df.columns:
            print(f"WARNING: 'rdkit_mol_cistrans_stereo' not found. Available columns: {df.columns.tolist()}")
            print("Please set the mol column name manually below.")
            # Try to auto-detect an rdkit mol column
            mol_cols = [c for c in df.columns if df[c].dropna().apply(
                lambda x: isinstance(x, rdkit.Chem.rdchem.Mol)).any()]
            if mol_cols:
                print(f"Auto-detected possible mol column(s): {mol_cols}")
                df = df.rename(columns={mol_cols[0]: 'rdkit_mol_cistrans_stereo'})
                print(f"Using '{mol_cols[0]}' as the mol column.")
            else:
                raise ValueError("Could not find any column containing rdkit Mol objects.")

        # --- Validate regression column ---
        if regression != '' and regression not in df.columns:
            raise ValueError(f"Regression target '{regression}' not found. Columns: {df.columns.tolist()}")

        return cls(df, regression=regression)

    def __init__(self, df, regression = ''):
        super(ChIRo, self).__init__()
        self.df = df
        self.regression = regression
    
    def embed_mol(self, mol):
        if isinstance(mol, rdkit.Chem.rdchem.Conformer):
            mol = mol.GetOwningMol()
            conformer = mol
        elif isinstance(mol, rdkit.Chem.rdchem.Mol):
            mol = mol
            conformer = mol.GetConformer()

        # Edge Index
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)
        
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = np.array([atom.GetAtomicNum() for atom in atoms]) # Z
        positions = np.array([conformer.GetAtomPosition(atom.GetIdx()) for atom in atoms]) # xyz positions
        
        return edge_index, node_features, positions # edge_index, Z, pos
        
    def process_mol(self, mol):
        edge_index, Z, pos = self.embed_mol(mol)
        data = torch_geometric.data.Data(x = torch.as_tensor(Z).unsqueeze(1), edge_index = torch.as_tensor(edge_index, dtype=torch.long))
        data.pos = torch.as_tensor(pos, dtype = torch.float)
        data.z = torch.as_tensor(Z)
        return data
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        mol = deepcopy(self.df.iloc[key].rdkit_mol_cistrans_stereo)
        data = self.process_mol(mol)
        
        if self.regression != '':
            data.y = torch.tensor(
                deepcopy(self.df.iloc[key][self.regression]),
                dtype=torch.float
            ).unsqueeze(0)
        
        return data
    

def get_chiro_dataset(pre_transform = None, force_subset=True):
    DATAPATH = 'data/raw_data/'
    train_df = ChIRo.load_data(
        path=os.path.join(DATAPATH, 'chiro', 'train_RS.pkl'), regression='RS_label_binary'
    )
    val_df = ChIRo.load_data(
        path=os.path.join(DATAPATH, 'chiro', 'validation_RS.pkl'), regression='RS_label_binary'
    )
    test_df = ChIRo.load_data(
        path=os.path.join(DATAPATH, 'chiro', 'test_RS.pkl'), regression='RS_label_binary'
    )

    if force_subset:
        train_df.df = train_df.df.iloc[:5000].reset_index(drop=True)
        val_df.df   = val_df.df.iloc[:200].reset_index(drop=True)
        test_df.df  = test_df.df.iloc[:200].reset_index(drop=True)

    return train_df, val_df, test_df, None