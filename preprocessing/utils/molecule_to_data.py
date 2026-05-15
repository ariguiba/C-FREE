import torch
from torch_geometric.data import Data

import numpy as np 

from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.rdchem import BondType

from preprocessing.utils.molecule_feats import atom_to_feature_vector, bond_to_feature_vector


NODE_MAPPING1 = {
    0: 6,  # C
    1: 8,  # O
    2: 7,  # N
    3: 9,  # F
    5: 16,  # S
    6: 17,  # Cl
    9: 35,  # Br
    15: 53,  # I
    16: 15,  # P
}

def zinc_data_to_mol_data(data):
    mol = RWMol()
    # Add atoms
    for i in range(data.num_nodes):
        atom_type_idx = data.x[i].argmax().item()  # if one-hot encoding
        atom = Chem.Atom(NODE_MAPPING1[atom_type_idx])  # or map index to element symbol
        mol.AddAtom(atom)

    # Add bonds
    edge_index = data.edge_index
    for src, dst in edge_index.t().tolist():  # PyG uses [2, num_edges]
        # You may need to know bond type; here we assume single
        if mol.GetBondBetweenAtoms(src, dst) is None:
            mol.AddBond(src, dst, BondType.SINGLE)

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print("Skipping molecule, failed sanitization:", e)
        mol = None

    molecule_data = mol_to_data_obj(mol, y=data.y)
    return molecule_data

def mol_to_data_obj(mol, y = None, simple=False, with_3d = False):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom, simple=simple))

    x = torch.tensor(np.asarray(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_feature = bond_to_feature_vector(bond, simple=simple)
            
            edge_attr.append(edge_feature)
            edge_attr.append(edge_feature)
        edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.asarray(edge_attr), dtype=torch.long)

    # coordinates
    if with_3d:
        pos = mol.GetConformer().GetPositions()
        pos = torch.from_numpy(pos).float()

    data = Data(x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                mol=mol, # This is needed to create the fragments
                **({'pos': pos} if with_3d and pos is not None else {}),
                **({'y': y} if y is not None else {})
            )

    return data
