import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip

from tqdm import tqdm
import pickle
from sklearn.utils import shuffle

from rdkit import Chem
torch.serialization.add_safe_globals([Chem.Mol])

from preprocessing.utils.molecule_to_data import mol_to_data_obj

RAW_DATA_FOLDER = 'data/raw_data/kraken/raw'

class MyKrakenDataset(InMemoryDataset):
    descriptors = ["sterimol_B5", "sterimol_L", "sterimol_burB5", "sterimol_burL"]

    def __init__(
        self,
        root,
        max_num_conformers=None,
        policy="original",
        full_name = "kraken_PLACEHODLER",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        simple=False,
        with_3d=False
    ):
        self.policy = policy
        self.full_name = full_name
        self.max_num_conformers = max_num_conformers
        self.simple = simple
        self.with_3d=with_3d
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.processed_paths[0])
        self.num_molecules = 1552  # hard-coded for now

    @property
    def processed_file_names(self):
        return f"{self.full_name}_processed.pt"

    @property
    def raw_file_names(self):
        return "Kraken.zip"
    
    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        data_list = []
        descriptors = self.descriptors
        with_mol = False if self.policy == "original" else True

        raw_zip_file = f"{RAW_DATA_FOLDER}/Kraken.zip"
        raw_pickle_file = f"{RAW_DATA_FOLDER}/Kraken.pickle"
        extract_zip(raw_zip_file, RAW_DATA_FOLDER)

        with open(raw_pickle_file, "rb") as f:
            kraken = pickle.load(f)

        ligand_ids = list(kraken.keys())
        y = []

        # Calculates max number of conformers 
        max_conformers = 0
        for ligand_id in tqdm(ligand_ids):
            _, _, conformer_dict = kraken[ligand_id]
            conformer_ids = list(conformer_dict.keys())
            n_conformers = len(conformer_ids)
            if n_conformers > max_conformers:
                max_conformers = n_conformers

        if self.max_num_conformers == "all":
            self.max_num_conformers = max_conformers
        
        print(f"Processing Kraken with max_n_conformers = {self.max_num_conformers} for policy: {self.policy}")
        # for ligand_id in tqdm(ligand_ids):
        for ligand_id in ligand_ids:
            positions = []

            smiles, boltz_avg_properties, conformer_dict = kraken[ligand_id]
            conformer_ids = list(conformer_dict.keys())

            if self.max_num_conformers is not None:
                # sort conformers by boltzmann weight and take the lowest energy conformers
                conformer_ids = sorted(
                    conformer_ids, key=lambda x: conformer_dict[x][1], reverse=True
                )
                conformer_ids = conformer_ids[: self.max_num_conformers]

            if len(conformer_ids) < self.max_num_conformers:
                repeats = (self.max_num_conformers // len(conformer_ids)) + 1
                conformer_ids = (conformer_ids * repeats)[: self.max_num_conformers]

            for conformer_id in conformer_ids:
                mol_sdf, _, _ = conformer_dict[conformer_id]
                mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False)
                data = mol_to_data_obj(mol, simple=self.simple, with_3d=self.with_3d)
                positions.append(data.pos)

            y = torch.tensor(
                [boltz_avg_properties[descriptor] for descriptor in descriptors]
            )

            if "_b5" in self.raw_paths[0]:
                y = y[0]
            elif "_l" in self.raw_paths[0]:
                y = y[1]
            elif "_burb5" in self.raw_paths[0]:
                y = y[2]
            elif "_burl" in self.raw_paths[0]:
                y = y[3]
            # else:
            #     raise NotImplementedError("Unknown dataset")
            
            data_full = Data(
                    x=data.x,
                    z=data.x[:, 0],
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    pos=positions,
                    y=y.unsqueeze(0),
                    smiles=smiles,
                    mol=data.mol if with_mol else None
                )

            if self.pre_transform is not None:
                transformed = self.pre_transform(data_full)
                if transformed is not None:
                    data_list.append(transformed)
            else:
                data_list.append(data_full)

        no_pos = 0
        for data in data_list:
            if not hasattr(data, "pos") or data.pos is None:
                no_pos +=1
                data.pos = [torch.zeros((data.x.shape[0], 3)) for _ in range(self.max_num_conformers)]  # dummy positions
        
        print(f"Warning!!! No positions available for {no_pos} Subgraph Objects. If you want to use 3D information, make sure to set with_3d=True in the config file.")
        self.save(data_list, self.processed_paths[0])

    def get_idx_split(self, train_ratio=0.8, valid_ratio=0.1, seed=123):
        molecule_ids = shuffle(range(self.num_molecules), random_state=seed)
        train_size = int(train_ratio * self.num_molecules)
        valid_size = int(valid_ratio * self.num_molecules)

        train_idx = torch.tensor(molecule_ids[:train_size])
        valid_idx = torch.tensor(molecule_ids[train_size : train_size + valid_size])
        test_idx = torch.tensor(molecule_ids[train_size + valid_size :])

        return train_idx, valid_idx, test_idx

def get_kraken(datapath, n_conformers, policy_name, full_name, pre_transform = None, simple = False, with_3d=False):
    kraken_dataset = MyKrakenDataset(
        root=datapath, 
        max_num_conformers=n_conformers, 
        policy = policy_name,
        full_name=full_name,
        transform=None,
        pre_transform=pre_transform,
        force_reload=False,
        simple=simple,
        with_3d=with_3d
    )

    train_idx, val_idx, test_idx = kraken_dataset.get_idx_split(
        train_ratio=0.8, valid_ratio=0.1, seed=123
    )

    train = kraken_dataset[train_idx]
    val = kraken_dataset[val_idx]
    test = kraken_dataset[test_idx]
    stds = train.y.std(dim=0, keepdim=True)
    mean = train.y.mean(dim=0, keepdim=True)

    return train, val, test, (mean, stds)