from torch_geometric.data import Data

import pandas as pd
import os
from functools import partial

from preprocessing.utils.molecule_to_data import mol_to_data_obj
from preprocessing.utils.scaffolds import get_scaffold_split, random_scaffold_split, get_random_splits
from torch_geometric.datasets import MoleculeNet

RAW_DATA_FOLDER = 'data/raw_data/moleculenet/raw' # Change this to where you have your raw data

DATASET_CONFIG = {
    "esol": {"name": "ESOL"},
    "freesolv": {"name": "FreeSolv"},
    "lipo": {"name": "Lipo"},
    "bace": {"name": "BACE", "filename": "bace", "column": "mol"},
    "bbbp": {"name": "BBBP", "filename": "BBBP", "column": "smiles"},
    "clintox": {"name": "ClinTox", "filename": "clintox", "column": "smiles"},
    "toxcast": {"name": "ToxCast", "filename": "toxcast_data", "column": "smiles"},
    "hiv": {"name": "HIV", "filename": "HIV", "column": "smiles"},
    "sider": {"name": "SIDER", "filename": "sider", "column": "smiles"},
    "tox21": {"name": "Tox21", "filename": "tox21", "column": "smiles"},
    "muv": {"name": "MUV", "filename": "muv", "column": "smiles"},
}

class MyMoleculeNet(MoleculeNet):
    def __init__(self, custom_raw_dir=RAW_DATA_FOLDER, **kwargs):
        self.custom_raw_dir = custom_raw_dir
        super().__init__(**kwargs)
    
    @property
    def raw_dir(self):
        if self.custom_raw_dir:
            return self.custom_raw_dir
        return super().raw_dir
    
    def download(self):
        # Check if raw files already exist
        if all(os.path.exists(os.path.join(self.raw_dir, f)) for f in self.raw_file_names):
            print(f"Raw files already exist in {self.raw_dir}, skipping download")
            return
        super().download()

def pre_filter(data):
        return data.edge_index.numel() > 0

def my_from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
    simple: bool = False,
    with_3d: bool=False
) -> Data:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    data = mol_to_data_obj(mol, simple=simple, with_3d=False)
    data.smiles = smiles
    return data

def get_moleculenet(dataset, datapath, pre_transform = None, with_3d=False, seed=0):
    config = DATASET_CONFIG.get(dataset.lower())
    if config is None:
        available = ", ".join(DATASET_CONFIG.keys())
        raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")
    
    dataset_name = config["name"]
    dataset_filename = config["filename"]
    column = config["column"]

    moleculenet_ds = MyMoleculeNet(
        root=datapath, 
        name=dataset_name, 
        transform=None,
        pre_filter=pre_filter,
        pre_transform=pre_transform,
        from_smiles=partial(my_from_smiles, simple=False, with_3d=with_3d)
    )
    
    # Fix scaffolds
    smiles = pd.read_csv( f"{RAW_DATA_FOLDER}/{dataset_filename}.csv")[column]
    # train_set, val_set, test_set = random_scaffold_split(moleculenet_ds, smiles.tolist(), seed=seed, return_smiles=False)
    train_set, val_set, test_set = get_scaffold_split(moleculenet_ds, smiles.tolist(), return_smiles=False, seed=seed)
    #get_random_splits(moleculenet_ds, train_ratio=0.8, valid_ratio=0.1, seed=seed)
    
    print(f"Train set size: {len(train_set)}; Val set size: {len(val_set)}; Test set size: {len(test_set)}")

    stds = train_set.y.std(dim=0, keepdim=True)
    mean = train_set.y.mean(dim=0, keepdim=True)

    return train_set, val_set, test_set, (stds, mean)


