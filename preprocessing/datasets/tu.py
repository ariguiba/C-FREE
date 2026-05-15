import os
from torch_geometric.datasets import TUDataset as TUDataset_
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch

# From MolMix
def get_tu(args):
    policy = args.policy if hasattr(args, "policy") else "original"
    datapath = os.path.join(args.data_path, f"{args.dataset.lower()}/{policy}")
    
    train = TUDataset_(
        root=datapath,
        name=args.dataset.lower(),
        # policy=policy,
        split="train",
        transform=None, 
        pre_transform=None, 
        force_reload=False
    )
    val = TUDataset_(
        root=datapath,
        name=args.dataset.lower(),
        split="val",
        transform=None, 
        pre_transform=None, 
        force_reload=False
    )
    test = TUDataset_(
        root=datapath,
        name=args.dataset.lower(),
        split="test",
        transform=None, 
        pre_transform=None, 
        force_reload=False
    )

    return train, val, test, (None, None)


# From the ESAN Code
# class TUDataset(TUDataset_):
#     def __init__(self, root: str, name: str,
#                  transform=None,
#                  pre_transform=None,
#                  pre_filter=None,
#                  use_node_attr: bool = False, use_edge_attr: bool = False,
#                  cleaned: bool = False):

#         super().__init__(root, name, transform, pre_transform, pre_filter,
#                          use_node_attr, use_edge_attr, cleaned)

#     @property
#     def num_tasks(self):
#         return 1 if self.name != "IMDB-MULTI" else 3

#     @property
#     def eval_metric(self):
#         return 'acc'

#     @property
#     def task_type(self):
#         return 'classification'

#     def download(self):
#         super().download()

#     def process(self):
#         super().process()

#     # ASSUMPTION: node_idx features for ego_nets_plus are prepended
#     @property
#     def num_node_labels(self):
#         if self.data.x is None:
#             return 0
#         num_added = 2 if isinstance(self.pre_transform, EgoNets) and self.pre_transform.add_node_idx else 0
#         for i in range(self.data.x.size(1) - num_added):
#             x = self.data.x[:, i + num_added:]
#             if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
#                 return self.data.x.size(1) - i
#         return 0

#     def separate_data(self, seed, fold_idx):
#         # code taken from GIN and adapted
#         # since we only consider train and valid, use valid as test

#         assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
#         skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

#         labels = self.data.y.numpy()
#         idx_list = []
#         for idx in skf.split(np.zeros(len(labels)), labels):
#             idx_list.append(idx)
#         train_idx, test_idx = idx_list[fold_idx]

#         return {'train': torch.tensor(train_idx), 'valid': torch.tensor(test_idx), 'test': torch.tensor(test_idx)}
