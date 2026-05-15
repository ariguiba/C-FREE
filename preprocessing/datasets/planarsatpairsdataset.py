import os
import pickle

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import ToUndirected

from preprocessing.utils.transforms import get_pretransform, get_transform
from torch.utils.data import random_split

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, extra_path, transform=None, pre_transform=None, pre_filter=None,
    policy="original"):
        self.extra_path = extra_path
        self.policy = policy
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return f'data_{self.policy}.pt'

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed_' + self.extra_path)

    @property
    def num_features(self):
        return self.data.num_node_features
    
    @property
    def num_node_features(self):
        return self.data.num_node_features
    
    @property
    def num_edge_features(self):
        return self.data.num_edge_features
    
    @property
    def num_classes(self):
        return int(torch.max(self.data.y)) + 1

    # def __getitem__(self, idx):
    #     return super().__getitem__(idx)
    
    def get(self, idx):
        data = super().get(idx)
        num_edges = data.edge_index.size(1)
        data.edge_attr = torch.zeros(num_edges, 1)
        return data

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(type(data_list[0]))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        

# def get_exp_dataset(args, force_subset, num_fold=10, subset_only=True):
#     extra_path = 'policy'
#     extra_path = extra_path if extra_path is not None else 'normal'
#     policy = args.policy if hasattr(args, "policy") else "original"
#     pre_transform = get_pretransform(args, extra_pretransforms=[ToUndirected()])
#     # transform = get_transform(args)

#     dataset = PlanarSATPairsDataset(os.path.join(args.data_path, args.dataset.upper()),
#                                     extra_path,
#                                     # transform=transform,
#                                     pre_transform=pre_transform,
#                                     policy=policy)
#     dataset._data.y = dataset._data.y.float()[:, None]
#     dataset._data.x = dataset._data.x.squeeze()

#     train_sets, val_sets, test_sets = [], [], []
#     for idx in range(num_fold):
#         train, val, test = separate_data(idx, dataset, num_fold)
#         train_set = dataset[train]
#         val_set = dataset[val]
#         test_set = dataset[test]

#         train_sets.append(train_set)
#         val_sets.append(val_set)
#         test_sets.append(test_set)

#     if subset_only:
#         train_sets = train_sets[0]
#         val_sets = val_sets[0]
#         test_sets = test_sets[0]
#         if force_subset:
#             train_sets = train_sets[:1]
#             val_sets = val_sets[:1]
#             test_sets = test_sets[:1]
#     else:
#         if force_subset:
#             train_sets = train_sets[0][:1]
#             val_sets = val_sets[0][:1]
#             test_sets = test_sets[0][:1]

#     return train_sets, val_sets, test_sets, None

def get_exp_dataset(args, force_subset, num_fold=10, subset_only=True):
    extra_path = 'policy'
    extra_path = extra_path if extra_path is not None else 'normal'
    policy = args.policy if hasattr(args, "policy") else "original"
    pre_transform = get_pretransform(args, extra_pretransforms=[ToUndirected()])
    #policy2transform(policy=policy, num_hops=1, k=2, process_subgraphs=lambda x: x)
    
    transform = get_transform(args)

    dataset = PlanarSATPairsDataset(os.path.join(args.data_path, args.dataset.upper()),
                                    extra_path,
                                    # transform=transform,
                                    pre_transform=pre_transform,
                                    policy=policy)
    dataset._data.y = dataset._data.y.float()[:, None]
    dataset._data.x = dataset._data.x.squeeze()

    # train, val, test = separate_data(0, dataset, 1)
    # train_set = dataset[train]
    # val_set = dataset[val]
    # test_set = dataset[test]

    # Compute lengths
    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len  # ensure it sums to total

    # Random split
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    return train_set.dataset, val_set.dataset, test_set.dataset, None