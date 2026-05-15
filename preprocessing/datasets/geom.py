import torch
from torch_geometric.data import Data, InMemoryDataset

import os
from os.path import join
from tqdm import tqdm
from sklearn.utils import shuffle
import json
import pickle
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
torch.serialization.add_safe_globals([Chem.Mol])

from preprocessing.utils.molecule_to_data import mol_to_data_obj

RAW_DATA_FOLDER = 'data/raw_data/geom/raw' # Change this to where you have your raw data

class MyGEOMDataset(InMemoryDataset):
    """Pre-Training 2D/3D GNN with GEOM"""

    def __init__(
        self,
        n_mol,
        n_conf,
        root,
        policy = "original",
        full_name = "geom_PLACEHOLDER",
        n_upper=9999,
        seed=777,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        empty=False,
        simple=False,
        with_3d=False
    ):
        # os.makedirs(root, exist_ok=True)
        # os.makedirs(join(root, "raw"), exist_ok=True)
        # os.makedirs(join(root, "processed"), exist_ok=True)

        self.num_molecules = 276518
        self.simple = simple
        self.policy = policy
        self.full_name = full_name
        self.with_3d=with_3d
        self.root, self.seed = root, seed
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(MyGEOMDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print(
            "root: {},\ndata: {},\nn_mol: {},\nn_conf: {}".format(
                self.root, self.data, self.n_mol, self.n_conf
            )
        )

    @property
    def raw_file_names(self):
        return os.listdir(RAW_DATA_FOLDER)

    @property
    def processed_file_names(self):
        return f"{self.full_name}_processed.pt"

    def download(self):
        return

    def process(self):
        data_list = []
        smiles_list = []

        geom_file = f"{RAW_DATA_FOLDER}/summary_geom.json"
        with open(geom_file, "r") as f:
            geom_summary = json.load(f)
        geom_summary = list(geom_summary.items())
        print("# SMILES: {}".format(len(geom_summary)))
        # expected: 304,466 molecules

        raw_mols_file = f"{RAW_DATA_FOLDER}/geom_molecule_objects.pkl"

        mol_idx, idx, notfound = 0, 0, 0

        if os.path.exists(raw_mols_file):
            print("Loading preprocessed molecules...")
            with open(raw_mols_file, "rb") as f:
                data_list = pickle.load(f)
        else: 
            print("Extracting molecules from the raw files and convert to PyG objects: ")
            for smiles, sub_dic in tqdm(geom_summary):
                # if idx > 1000:
                #     break
                """path in json and sub directory should match: pickle file contains the conformers"""
                if sub_dic.get("pickle_path", "") == "":
                    notfound += 1
                    continue

                mol_path = join(RAW_DATA_FOLDER, sub_dic["pickle_path"])
                with open(mol_path, "rb") as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic["conformers"]

                    """ energy should ascend, which turns out it doesn't"""
                    energy_list = [item["relativeenergy"] for item in conformer_list]
                    # assert np.all(np.diff(energy_list) >= 0)
                    conformer_list = [
                        conformer_list[i] for i in np.argsort(energy_list)
                    ]

                    """ count should match """
                    # there are other ways (e.g. repeated sampling) for molecules that do not have enough conformers
                    conf_n = len(conformer_list)
                    # if conf_n < self.n_conf or conf_n > self.n_upper:
                    if conf_n < self.n_conf:
                        notfound += 1
                        continue

                    """ SMILES should match 
                    This ensures that the SMILES is valid and roundtrip-convertible 
                    (no weird stereochemistry loss, parsing errors, or RDKit quirks)"""
                    # Ref:
                    # https://github.com/learningmatter-mit/geom/issues/4#issuecomment-853486681
                    # https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb
                    conf_list = [
                        AllChem.MolToSmiles(m2)
                        for rd in conformer_list[:self.n_conf]
                        if (m := rd["rd_mol"]) is not None
                        if (m2 := AllChem.MolFromSmiles(AllChem.MolToSmiles(m))) is not None
                    ]

                    conf_list_raw = [
                        AllChem.MolToSmiles(rd_mol["rd_mol"])
                        for rd_mol in conformer_list[: self.n_conf]
                    ]
                    # check that they're all the same
                    same_confs = len(list(set(conf_list))) == 1
                    same_confs_raw = len(list(set(conf_list_raw))) == 1
                    if not same_confs:
                        if same_confs_raw is True:
                            print("Interesting")
                        notfound += 1
                        continue
                    
                    """Extracting Conformers """
                    positions = []
                    for conformer_dict in conformer_list[: self.n_conf]:
                        rdkit_mol = conformer_dict["rd_mol"]
                        data = mol_to_data_obj(rdkit_mol, simple=self.simple, with_3d=self.with_3d)
                        
                        data.name = torch.tensor([mol_idx])
                        data.id = torch.tensor([idx])
                        data.smiles = smiles
                        positions.append(data.pos)
                        # smiles_list.append(smiles)        
                        idx += 1

                    data_full = Data(
                        x=data.x,
                        z=data.x[:, 0],
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                        pos=positions,
                        # y=y.unsqueeze(0),
                        smiles=data.smiles,
                        mol=data.mol
                        # mol=data.mol if with_mol else None
                    )
                data_list.append(data_full)

                if mol_idx + 1 >= self.n_mol:
                    break
                if same_confs:
                    mol_idx += 1

            with open(raw_mols_file, "wb") as f:
                pickle.dump(data_list, f)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list, desc="Applying pre_transform")]

        no_pos = 0
        for data in data_list:
            if not hasattr(data, "pos") or data.pos is None:
                no_pos +=1
                data.pos = [torch.zeros((data.x.shape[0], 3)) for _ in range(self.n_conf)]  # dummy positions
        
        print(f"Warning!!! No positions available for {no_pos} Subgraph Objects. If you want to use 3D information, make sure to set with_3d=True in the config file.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers have been processed" % idx)
    
    def get_idx_split(self, train_ratio=0.8, valid_ratio=0.1, seed=123):
        molecule_ids = shuffle(range(self.num_molecules), random_state=seed)
        train_size = int(train_ratio * self.num_molecules)
        valid_size = int(valid_ratio * self.num_molecules)

        train_idx = torch.tensor(molecule_ids[:train_size])
        valid_idx = torch.tensor(molecule_ids[train_size : train_size + valid_size])
        test_idx = torch.tensor(molecule_ids[train_size + valid_size :])

        return train_idx, valid_idx, test_idx

def get_geom(datapath, n_conformers, policy_name, full_name, pre_transform = None, simple = False, with_3d=False):
    geom_dataset = MyGEOMDataset(
        n_mol=276518, #298287,
        n_conf=n_conformers,
        root=datapath,
        policy = policy_name,
        full_name = full_name,
        transform=None,
        pre_transform=pre_transform,
        simple=simple,
        with_3d=with_3d
    )
    
    train_idx, val_idx, test_idx = geom_dataset.get_idx_split(
        train_ratio=0.8, valid_ratio=0.1, seed=123
    )

    train = geom_dataset[train_idx]
    val = geom_dataset[val_idx]
    test = geom_dataset[test_idx]

    return train, val, test, (None, None)