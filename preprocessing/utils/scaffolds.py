from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
from itertools import compress
import torch
from rdkit import Chem
from sklearn.utils import shuffle
import random
from torch.utils.data import Subset

def bm_scaffold(smi: str, include_chirality: bool = False) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None: return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=include_chirality)

def get_scaffold_split(dataset, smiles_list, task_idx=None, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=20, return_smiles=False):
    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != 0
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    groups = defaultdict(list)
    for i, (k, s) in enumerate(smiles_list):
        groups[bm_scaffold(s)].append(i)
    rng = random.Random(seed)
    bins = list(groups.values())
    rng.shuffle(bins)                 # stable tiebreak
    bins.sort(key=len, reverse=True)  # big scaffolds first

    n = len(smiles_list)
    n_train = int(round(frac_train*n))
    n_valid = int(round(frac_valid*n))
    n_test  = n - n_train - n_valid

    train, valid, test = [], [], []
    for b in bins:
        if len(train) + len(b) <= n_train: train += b
        elif len(valid) + len(b) <= n_valid: valid += b
        else: test += b

    def trim_to(lst, k): return lst[:k], lst[k:]
    train, spill = trim_to(train, n_train); valid += spill
    valid, spill = trim_to(valid, n_valid); test  += spill

    #Subset(dataset, train), Subset(dataset, valid), Subset(dataset, test),  
    return dataset[train], dataset[valid], dataset[test]

def generate_scaffold(smiles, include_chirality=False):
    """Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold"""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold

def get_scaffold_split_old(
    dataset,
    smiles_list,
    task_idx=None,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    return_smiles=False,
):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
    :param frac_train, frac_valid, frac_test: fractions
    :param return_smiles: return SMILES if Ture
    :return: train, valid, test slices of the input dataset obj."""
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        try:
            scaffold = bm_scaffold(smiles, include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)
        except ValueError:
            pass

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            (train_smiles, valid_smiles, test_smiles),
        )

def random_scaffold_split(
    dataset,
    smiles_list,
    task_idx=None,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=0,
    return_smiles=False,
):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/
        chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
    :param frac_train, frac_valid, frac_test: fractions, floats
    :param seed: seed
    :param return_smiles: whether return smiles
    :return: train, valid, test slices of the input dataset obj"""

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = generate_scaffold(smiles, include_chirality=True)
            scaffolds[scaffold].append(ind)
        else:
            continue

    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)

    #scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            (train_smiles, valid_smiles, test_smiles),
        )

def get_random_splits(molecule_ds, train_ratio=0.8, valid_ratio=0.1, seed=123):
    length = len(molecule_ds)
    molecule_ids = shuffle(range(length), random_state=seed)
    train_size = int(train_ratio * length)
    valid_size = int(valid_ratio * length)

    train_idx = torch.tensor(molecule_ids[:train_size])
    val_idx = torch.tensor(molecule_ids[train_size : train_size + valid_size])
    test_idx = torch.tensor(molecule_ids[train_size + valid_size :])

    train = molecule_ds[train_idx]
    val = molecule_ds[val_idx]
    test = molecule_ds[test_idx]

    return train, val, test