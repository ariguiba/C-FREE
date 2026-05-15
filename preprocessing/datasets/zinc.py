import os
from torch_geometric.datasets import ZINC
from preprocessing.utils.molecule_to_data import zinc_data_to_mol_data

def get_zinc(datapath, pre_transform): #args):
    is_subset = True
    datapath = os.path.join(datapath, 'zinc-subset'  if is_subset else 'zinc-full')
    
    # datasettype = "full/" if args.dataset == "full" else "subset/"
    # is_subset = False if args.dataset == "zinc-full" else True
    # policy = args.policy if hasattr(args, "policy") else "original"
    # extra = "full"
    # datapath = os.path.join(args.data_path, 'zinc-' + datasettype + policy + extra)

    # train_set = ZINC(datapath, split="train", subset=is_subset, pre_filter=None, transform=None, pre_transform=zinc_data_to_mol_data, force_reload=False)
    # val_set = ZINC(datapath, split="val", subset=is_subset, pre_filter=None, transform=None, pre_transform=zinc_data_to_mol_data, force_reload=False)
    # test_set = ZINC(datapath, split="test", subset=is_subset,pre_filter=None, transform=None, pre_transform=zinc_data_to_mol_data, force_reload=False)

    train_set = ZINC(datapath, split="train", subset=is_subset, pre_filter=None, transform=None, pre_transform=pre_transform, force_reload=False)
    val_set = ZINC(datapath, split="val", subset=is_subset, pre_filter=None, transform=None, pre_transform=pre_transform, force_reload=False)
    test_set = ZINC(datapath, split="test", subset=is_subset,pre_filter=None, transform=None, pre_transform=pre_transform, force_reload=False)

    for sp in [train_set, val_set, test_set]:
        sp._data.x = sp._data.x.squeeze()
        sp._data.edge_attr = sp._data.edge_attr.squeeze()
        sp._data.y = sp._data.y[:, None]
    
    print(f'Train set: {len(train_set)}, Val set: {len(val_set)}, Test set: {len(test_set)}')
    stds = train_set.y.std(dim=0, keepdim=True)
    mean = train_set.y.mean(dim=0, keepdim=True)
    return train_set, val_set, test_set, (stds, mean)