import torch
import torch_geometric
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.loader import PrefetchLoader
from torch.utils.data import SubsetRandomSampler, random_split
from collections import namedtuple
import os

torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])
torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])

from utils.masking import process_sugraph_separated, process_sugraph_together
from preprocessing.datasets.chiro import get_chiro_dataset
from preprocessing.datasets.planarsatpairsdataset import get_exp_dataset
from preprocessing.datasets.marcel.kraken import get_kraken
from preprocessing.datasets.marcel.drugs import get_drugs
from preprocessing.datasets.geom import get_geom
from preprocessing.datasets.moleculenet import get_moleculenet
from preprocessing.datasets.zinc import get_zinc
from preprocessing.datasets.spice import get_spice
from preprocessing.datasets.qm9 import get_qm9

from preprocessing.utils.transforms import get_pretransform
from preprocessing.subgraphs import create_subgraph_config_name

NUM_WORKERS = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = "data"


AttributedDataLoader = namedtuple(
    "AttributedDataLoader",
    [
        "loader",
        "std",
        "task",
    ],
)

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

def get_dataset(dataset, datapath, n_conformers, policy_name, full_name, pre_transform = None, simple = False, with_3d=False, args=None): 
    task = "graph"

    if dataset.startswith("zinc"):
        train_set, val_set, test_set, std = get_zinc(datapath, pre_transform=pre_transform)
    elif dataset.startswith("chiro"):
        train_set, val_set, test_set, std = get_chiro_dataset(pre_transform=pre_transform)
    elif dataset.startswith("spice"):
        train_set, val_set, test_set, std = get_spice(datapath, pre_transform=pre_transform)
    elif dataset.startswith("qm9"):
        train_set, val_set, test_set, std = get_qm9(datapath, pre_transform=pre_transform)
    # elif dataset in ["proteins", "mutag"]:
    #     train_set, val_set, test_set, std = get_tu(args)
    elif dataset in ["esol", "freesolv", "lipo", "bace", "bbbp", "toxcast", "clintox", "hiv", "tox21", "sider", "muv"]:
        seed = getattr(args, "seed", 0)
        train_set, val_set, test_set, std = get_moleculenet(dataset, datapath, pre_transform = pre_transform, with_3d=with_3d, seed=seed)
    elif dataset.startswith("kraken"):
        train_set, val_set, test_set, std = get_kraken(datapath, n_conformers, policy_name, full_name, pre_transform = pre_transform, simple = simple, with_3d=with_3d)
    elif dataset.startswith("drugs"):
        train_set, val_set, test_set, std = get_drugs(datapath, n_conformers, policy_name, full_name, pre_transform = pre_transform, simple = simple, with_3d=with_3d)
    elif dataset.startswith("geom"):
        train_set, val_set, test_set, std = get_geom(datapath, n_conformers, policy_name, full_name, pre_transform = pre_transform, simple = simple, with_3d=with_3d)
    elif dataset.startswith("exp"):
        train_set, val_set, test_set, std = get_exp_dataset(args, force_subset=False, subset_only=True)
    
    else:
        raise NotImplementedError
    
    return train_set, val_set, test_set, std, task

def get_full_name(n_conformers=None, policy="original", dataset_base="kraken", simple=False, with_3d=False):
    base = dataset_base
    if simple:
        base += "_simple"
    if with_3d:
        base += f"_3d"
    if n_conformers is not None:
        base += f"_{n_conformers}" 
    return f"{base}_{policy}"

def get_data(args, dataset=None, is_probe = False):
    base = args.probe if is_probe else args
    data_base = base.data
    dataset = data_base.dataset.lower() if dataset is None else dataset

    n_conformers = getattr(data_base, "n_conformers", 1)
    with_3d = getattr(data_base, "with_3d", False)
    policy = getattr(data_base, "policy", "original")
    policy_name = create_subgraph_config_name(data_base.subgraph_types, data_base.num_samples) if policy != "original" else "original"
    simple = getattr(data_base, "simple", False)

    full_name = get_full_name(n_conformers, policy_name, dataset, simple, with_3d)
    datapath = os.path.join(f"{args.data_path}/{dataset}/{full_name}")
    pre_transform = get_pretransform(data_base, dataset)

    batch_size = base.batch_size        
    train_set, val_set, test_set, std, task = get_dataset(dataset, datapath, n_conformers, policy_name, full_name, pre_transform, simple, with_3d, args) 

    # Take only p percent of the training data
    # seed = args.seed if hasattr(args, "seed") else 711
    split_seed = 12345
    p = data_base.train_subset_perc if hasattr(data_base, "train_subset_perc") else 1.0
    torch.manual_seed(split_seed)
    train_size = int(p * len(train_set))
    rest_size = len(train_set) - train_size
    train_set, _ = random_split(train_set, [train_size, rest_size], generator=torch.Generator().manual_seed(split_seed))

    if policy == "original":
        FOLLOW_BATCH = ["x"]
    else:
        FOLLOW_BATCH = ["x", "subgraph_idx", "mol_idx"]

    if not dataset.lower().startswith("zinc") and not dataset.lower().startswith("exp"): # ZINC doesn't contain 3d information
        FOLLOW_BATCH.append("z")
    
    if not is_probe and args.ssl:
        print("Creating Context/Target Batches ...")
        
        # Option 1: separated - return each subgraph as one Data object
        # Option 2: together - each Data object contains subgraphs of same molecule
        process_together = args.data.process_together if hasattr(args, "data") and hasattr(args.data, "process_together") else True
        if process_together: 
            processed_pairs = process_sugraph_together(train_set, with_3d=with_3d)
            train_set = PreprocessedDataset(processed_pairs)
            processed_pairs_eval = process_sugraph_together(val_set, with_3d=with_3d)
            val_set = PreprocessedDataset(processed_pairs_eval)
        else:
            processed_pairs = process_sugraph_separated(train_set, with_3d=with_3d)
            train_set = PreprocessedDataset(processed_pairs)
            processed_pairs_eval = process_sugraph_separated(val_set, with_3d=with_3d)
            val_set = PreprocessedDataset(processed_pairs_eval)
        print("Subgraph pairs generation done :) ")
    
    if is_probe or not args.ssl:
        trn_loader = AttributedDataLoader(
            loader=PyGDataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=not args.debug,
                num_workers=0,
                drop_last=False,
                follow_batch=FOLLOW_BATCH,
                # sampler=sampler
            ),
            std=std,
            task=task,
        )

        val_loader = AttributedDataLoader(
            loader=PyGDataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                follow_batch=FOLLOW_BATCH,
            ),
            std=std,
            task=task,
        )
        tst_loader = AttributedDataLoader(
            loader=PyGDataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                follow_batch=FOLLOW_BATCH,
            ),
            std=std,
            task=task,
        )
    else:
        # batch_sample_ratio = args.data.batch_sample_ratio if hasattr(args, "data") and hasattr(args.data, "batch_sample_ratio") else 0.5
        # indices_train = np.random.choice(len(train_set), int(len(train_set)* batch_sample_ratio), replace=False)
        # trn_sampler = SubsetRandomSampler(indices_train)

        # indices_val = np.random.choice(len(val_set), int(len(val_set)* batch_sample_ratio), replace=False)
        # val_sampler = SubsetRandomSampler(indices_val)

        trn_loader = PrefetchLoader(
            loader=PyGDataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=not args.debug,
                num_workers=NUM_WORKERS,
                drop_last=False,
                follow_batch=FOLLOW_BATCH,
                pin_memory=False,
                prefetch_factor=5,
                persistent_workers=True
                # sampler=trn_sampler
            ),
            # std=std,
            # task=task,
        )

        val_loader = PrefetchLoader(
            loader=PyGDataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                follow_batch=FOLLOW_BATCH,
                pin_memory=False,
                prefetch_factor=5,
                persistent_workers=True
                # sampler=val_sampler
            ),
            # std=std,
            # task=task,
        )

        tst_loader = PrefetchLoader(
            loader=PyGDataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                follow_batch=FOLLOW_BATCH,
                pin_memory=False,
                prefetch_factor=5,
                persistent_workers=True
                # sampler=sampler
            ),
            # std=std,
            # task=task,
        )


    return trn_loader, val_loader, tst_loader, task
     

if __name__ == "__main__":
    pass