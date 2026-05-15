import os
import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from sklearn.utils import shuffle
import pickle

from preprocessing.utils.molecule_to_data import mol_to_data_obj

RAW_DATA_FOLDER = 'data/raw_data/drugs/raw' # Change this to where you have your raw data

class MyDrugsDataset(InMemoryDataset):
    descriptors = ["energy", "ip", "ea", "chi"]

    def __init__(
        self,
        root,
        max_num_conformers=None,
        policy = "original",
        full_name = "drugs_PLACEHOLDER",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        simple=False,
        with_3d=False
    ):
        self.max_num_conformers = max_num_conformers
        self.policy = policy
        self.simple = simple
        self.full_name = full_name
        self.with_3d = with_3d
        
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.processed_paths[0])
        self.num_molecules = 75099  # hard-coded for now
        

    @property
    def processed_file_names(self):
        return f"{self.full_name}_processed.pt"

    @property
    def raw_file_names(self):
        return "Drugs.zip"

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        quantities = self.descriptors
        with_mol = False if self.policy == "original" else True

        mols = defaultdict(list)

        raw_zip_file = f"{RAW_DATA_FOLDER}/Drugs.zip"
        raw_sdf_file = f"{RAW_DATA_FOLDER}/Drugs.sdf"
        extract_zip(raw_zip_file, RAW_DATA_FOLDER)

        raw_mols_file = f"{RAW_DATA_FOLDER}/molecule_objects.pkl"

        if os.path.exists(raw_mols_file):
            print("Loading preprocessed molecules...")
            with open(raw_mols_file, "rb") as f:
                mols = pickle.load(f)
        else:   
            print("Extracting molecules from the sdf file and convert to PyG objects: ")
            with Chem.SDMolSupplier(raw_sdf_file, removeHs=False) as suppl:
                for idx, mol in enumerate(tqdm(suppl)):
                    if mol is None:
                        continue
                    id_ = mol.GetProp("ID")
                    name = mol.GetProp("_Name")
                    smiles = mol.GetProp("smiles")
                
                    data = mol_to_data_obj(mol, simple=self.simple, with_3d=self.with_3d)
                    
                    data.name = name
                    data.id = id_
                    data.smiles = smiles
                    data.y = []

                    for quantity in quantities:
                        data.y.append(float(mol.GetProp(quantity)))
                    data.y = torch.Tensor(data.y).unsqueeze(0)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    
                    mols[name].append(data)

            with open(raw_mols_file, "wb") as f:
                pickle.dump(mols, f)

        label_file = raw_sdf_file.replace(".sdf", ".csv")
        labels = pd.read_csv(label_file)

        print(f"Processing Drugs with max_n_conformers = {self.max_num_conformers} for policy: {self.policy}")
        data_list = []
        # for name, mol_list in tqdm(mols.items()):
        for name, mol_list in mols.items():
            row = labels[labels["name"] == name]

            y = torch.Tensor([row[quantity].item() for quantity in quantities])

            if "drugs_energy" in self.raw_paths[0]:
                y = y[0]
            elif "drugs_ip" in self.raw_paths[0]:
                y = y[1]
            elif "drugs_ea" in self.raw_paths[0]:
                y = y[2]
            elif "drugs_chi" in self.raw_paths[0]:
                y = y[3]
            # else:
            #     raise NotImplementedError("Unknown dataset")

            if self.max_num_conformers is not None:
                # sort energy and take the lowest energy conformers
                mol_list = sorted(
                    mol_list, key=lambda x: x.y[:, quantities.index("energy")].item()
                )
                mol_list = mol_list[: self.max_num_conformers]

            if len(mol_list) < self.max_num_conformers:
                repeats = (self.max_num_conformers // len(mol_list)) + 1
                mol_list = (mol_list * repeats)[: self.max_num_conformers]

            data = mol_list[0]
            data_full = Data(
                    x=data.x,
                    z=data.x[:, 0],
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    pos=[mol.pos for mol in mol_list],
                    y=y.unsqueeze(0),
                    smiles=data.smiles,
                    mol=data.mol if with_mol else None
                )
            
            if self.pre_transform is not None:
                transformed = self.pre_transform(data_full)
                if transformed is not None:
                    data_list.append(transformed)
            else:
                data_list.append(data_full)

        self.save(data_list, self.processed_paths[0])

    def get_idx_split(self, train_ratio=0.8, valid_ratio=0.1, seed=123):
        molecule_ids = shuffle(range(self.num_molecules), random_state=seed)
        train_size = int(train_ratio * self.num_molecules)
        valid_size = int(valid_ratio * self.num_molecules)

        train_idx = torch.tensor(molecule_ids[:train_size])
        valid_idx = torch.tensor(molecule_ids[train_size : train_size + valid_size])
        test_idx = torch.tensor(molecule_ids[train_size + valid_size :])

        return train_idx, valid_idx, test_idx

def get_drugs(datapath, n_conformers, policy_name, full_name, pre_transform = None, simple = False, with_3d=False):
    drugs_dataset = MyDrugsDataset(
        root=datapath, 
        max_num_conformers=n_conformers, 
        policy = policy_name,
        full_name = full_name,
        transform=None,
        pre_transform=pre_transform,
        force_reload=False,
        simple=simple,
        with_3d=with_3d
    )

    train_idx, val_idx, test_idx = drugs_dataset.get_idx_split(
        train_ratio=0.8, valid_ratio=0.1, seed=123
    )

    train = drugs_dataset[train_idx]
    val = drugs_dataset[val_idx]
    test = drugs_dataset[test_idx]
    stds = train.y.std(dim=0, keepdim=True)
    mean = train.y.mean(dim=0, keepdim=True)

    return train, val, test, (mean, stds)