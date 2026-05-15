import os
from torch_geometric.datasets import QM9

from torch.utils.data import random_split
import torch

task_metainfo = {
    "qm9dft_v2_mu": {
        "mean": 2.7060374694700675,
        "std": 1.530388280934567,
        "target_name": "mu",
    },
    "qm9dft_v2_alpha": {
        "mean": 75.19129618702617,
        "std": 8.187762224050584,
        "target_name": "alpha",
    },
    "qm9dft_v2": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
    "qm9dft_v2_r2": {
        "mean": 1189.5274499667628,
        "std": 279.7561272394077,
        "target_name": "r2",
    },
    "qm9dft_v2_zpve": {
        "mean": 0.14852438909511897,
        "std": 0.03327377213900081,
        "target_name": "zpve",
    },
    "qm9dft_v2_u": {
        "mean": -1761.4806474164875,
        "std":  241.435201920648,
        "target_name": "u",
    },
    "qm9dft_v2_u0": {
        "mean": -1750.8129967646425 ,
        "std": 239.3124800445088,
        "target_name": "u0",
    },
    "qm9dft_v2_h": {
        "mean": -1771.5469283884158,
        "std": 243.1501571556723,
        "target_name": "h",
    },
    "qm9dft_v2_g": {
        "mean": -1629.388193917963,
        "std": 220.20626683425812,
        "target_name": "g",
    },
    "qm9dft_v2_cv": {
        "mean": 31.600675893490678,
        "std": 4.062456253369289,
        "target_name": "cv",
    },
}

def get_qm9(datapath, pre_transform): #args):
    
    dataset = QM9(datapath, pre_filter=None, transform=None, pre_transform=pre_transform, force_reload=False)

    # QM9 y indices mapping (from PyG QM9 dataset documentation)
    # y[:, 0] = mu (dipole moment)
    # y[:, 1] = alpha (polarizability)
    # y[:, 2] = homo
    # y[:, 3] = lumo
    # y[:, 4] = gap
    # y[:, 5] = r2 (electronic spatial extent)
    # y[:, 6] = zpve (zero point vibrational energy)
    # y[:, 7] = u0 (internal energy at 0K)
    # y[:, 8] = u (internal energy at 298K)
    # y[:, 9] = h (enthalpy at 298K)
    # y[:, 10] = g (free energy at 298K)
    # y[:, 11] = cv (heat capacity at 298K)

    # Extract means and stds for the 11 targets we're using
    means = []
    stds = []

    # Build in order of y indices
    means.append(task_metainfo["qm9dft_v2_mu"]["mean"])          # y[:, 0]
    stds.append(task_metainfo["qm9dft_v2_mu"]["std"])

    means.append(task_metainfo["qm9dft_v2_alpha"]["mean"])       # y[:, 1]
    stds.append(task_metainfo["qm9dft_v2_alpha"]["std"])

    means.extend(task_metainfo["qm9dft_v2"]["mean"])             # y[:, 2:5] (homo, lumo, gap)
    stds.extend(task_metainfo["qm9dft_v2"]["std"])

    means.append(task_metainfo["qm9dft_v2_r2"]["mean"])          # y[:, 5]
    stds.append(task_metainfo["qm9dft_v2_r2"]["std"])

    means.append(task_metainfo["qm9dft_v2_zpve"]["mean"])        # y[:, 6]
    stds.append(task_metainfo["qm9dft_v2_zpve"]["std"])

    means.append(task_metainfo["qm9dft_v2_u0"]["mean"])          # y[:, 7]
    stds.append(task_metainfo["qm9dft_v2_u0"]["std"])

    means.append(task_metainfo["qm9dft_v2_u"]["mean"])           # y[:, 8]
    stds.append(task_metainfo["qm9dft_v2_u"]["std"])

    means.append(task_metainfo["qm9dft_v2_h"]["mean"])           # y[:, 9]
    stds.append(task_metainfo["qm9dft_v2_h"]["std"])

    means.append(task_metainfo["qm9dft_v2_g"]["mean"])           # y[:, 10]
    stds.append(task_metainfo["qm9dft_v2_g"]["std"])

    means.append(task_metainfo["qm9dft_v2_cv"]["mean"])          # y[:, 11]
    stds.append(task_metainfo["qm9dft_v2_cv"]["std"])

    # Convert to tensors (or numpy arrays)
    means_tensor = torch.tensor(means, dtype=torch.float32)
    stds_tensor = torch.tensor(stds, dtype=torch.float32)

    # Or as tuple of arrays
    stats = (means_tensor, stds_tensor)


    # for sp in [train_set, val_set, test_set]:
    #     sp._data.x = sp._data.x.squeeze()
    #     sp._data.edge_attr = sp._data.edge_attr.squeeze()
    #     sp._data.y = sp._data.y[:, None]
    
    # print(f'Train set: {len(train_set)}, Val set: {len(val_set)}, Test set: {len(test_set)}')
    # stds = train_set.y.std(dim=0, keepdim=True)
    # mean = train_set.y.mean(dim=0, keepdim=True)
    # Compute lengths
    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len  # ensure it sums to total

    # Random split
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    return train_set, val_set, test_set, stats
    # return train_set, val_set, test_set, (stds, mean)