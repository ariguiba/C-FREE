
from typing import List, Optional
from utils.misc import Config
from tqdm import tqdm

from torch_geometric.transforms import AddRandomWalkPE, AddRemainingSelfLoops, Compose, ToUndirected

from preprocessing.subgraphs import policy2transform, SubgraphMix, Original
from preprocessing.utils.augment import AddLaplacianEigenvectorPE, AugmentWithPartition, AugmentWithDumbAttr, RenameLabel
from preprocessing.utils.augment_conformers import AugmentWithConformers, PygWithConformers


PRETRANSFORM_PRIORITY = {
    AugmentWithConformers: 99,
    PygWithConformers: 100,
    ToUndirected: 102,
    AddRemainingSelfLoops: 100,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithPartition: 98,
    AugmentWithDumbAttr: 98,
    RenameLabel: 0,
    Original: 1,
    SubgraphMix: 1,
}

num_graphs = {
        'zinc': 12001, #[1000, 1000],
        'exp': 1200,
        'kraken': 1552,
        'kraken_b5': 1552,
        'kraken_l': 1552,
        'kraken_burb5': 1552,
        'kraken_burl': 1552,
        'drugs': 75099,
        'drugs_ip': 75099,
        'drugs_ea': 75099,
        'drugs_chi': 75099,
        'bbbp': 2050,
        'geom': 304318,
        'bace': 1513,
        'tox21': 7831,
        'bbbp': 2050 , 
        'clintox': 1484 ,
        'toxcast': 8597 ,
        'hiv': 41127 ,
        'sider': 1427,
        'muv': 93087,
        'spice': 1110165,
        'qm9': 133885,
        'chiro': 326866,
    }


def get_transform(transformations = None):
    if transformations is not None:
        return Compose(transformations)
    else:
        return None


def get_pretransform(args: Config, dataset = None, extra_pretransforms: Optional[List] = None):
    pretransform = []
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args, "rwse_dim") or hasattr(args, "lap_dim"):
        if hasattr(args, "rwse_dim"):
            pretransform.append(AddRandomWalkPE(20, "pestat_RWSE"))
        if hasattr(args, "lap_dim"):
            pretransform.append(AddLaplacianEigenvectorPE(4, is_undirected=True))
    
    # I think these are unecessary - we do this when we create the data objects already

    if hasattr(args, "with_3d") and args.with_3d and hasattr(args, "n_conformers_pyg"):
        pretransform.append(AugmentWithConformers(num_conformers=args.n_conformers_pyg))
        pretransform.append(ToUndirected())
    # if hasattr(args, "n_conformers_pyg"):
    #     pretransform.append(PygWithConformers(num_conformers=args.n_conformers_pyg))

    # 🔹 Add policy2transform integration
    total_num_graphs = num_graphs[dataset.lower()]
    pretransform.append(
        policy2transform(
            policy=getattr(args, "policy", "original"),
            subgraph_types=getattr(args, "subgraph_types", ['2-ego']),
            num_samples=getattr(args, "num_samples", [2]),
            process_subgraphs=lambda x: x,
            pbar=iter(tqdm(range(total_num_graphs))),
            with_3d=getattr(args, "with_3d", False),
        )
    )

    if pretransform:
        pretransform = sorted(
            pretransform, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True
        )
        return Compose(pretransform)
    else:
        return None


