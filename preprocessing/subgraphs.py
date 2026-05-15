from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph

from preprocessing.utils.create_subgraphs import create_ego_pairs, create_fragment_pairs, create_full_pairs, create_murcko_pairs

def preprocess(dataset, transform):
    def unbatch_subgraphs(data):
        subgraphs = []
        subgraph_ids = data.subgraph_idx.shape[0]
        for subgraph_id in range(subgraph_ids):
            # Get node indices for this subgraph
            node_mask = data.subgraph_batch == subgraph_id
            node_idx = node_mask.nonzero(as_tuple=False).view(-1)

            # Extract subgraph edges (edges where both ends are in this subgraph)
            edge_index, edge_attr = subgraph(
                node_idx,
                data.edge_index,
                data.edge_attr,
                relabel_nodes=True,
            )

            subgraphs.append(
                Data(
                    x = data.x[node_idx],
                    edge_index = edge_index,
                    edge_attr = edge_attr,
                    subgraph_idx = subgraph_id,
                )
            )
        return Data(x=subgraphs[0].x, y=data.y,
                    subgraphs=subgraphs)

    data_list = [unbatch_subgraphs(data) for data in dataset]

    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)
    dataset.transform = transform
    return dataset

class SubgraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return super().__inc__(key, value, *args, **kwargs)
        
class Graph2Subgraph:
    def __init__(self, process_subgraphs=lambda x: x, pbar=None, with_3d=False):
        self.with_3d = with_3d
        self.process_subgraphs = process_subgraphs
        self.pbar = pbar

    def __call__(self, data):
        assert data.is_undirected()

        subgraphs = self.to_subgraphs(data)
        subgraphs = [self.process_subgraphs(s) for s in subgraphs]
        
        batch = Batch.from_data_list(subgraphs)

        if self.pbar is not None: next(self.pbar)

        return SubgraphData(x=batch.x, 
                edge_index=batch.edge_index, 
                edge_attr=batch.edge_attr,
                subgraph_batch=batch.batch,
                y=data.y, 
                subgraph_idx=batch.subgraph_idx if hasattr(batch, "subgraph_idx") else 0,
                subgraph_group_id_batch = batch.subgraph_group_id if hasattr(batch, "subgraph_group_id") else (batch.subgraph_idx if hasattr(batch, "subgraph_idx") else 0), #subgraph_node_idx=batch.subgraph_node_idx,)
                num_subgraphs=len(subgraphs),
                num_nodes = data.num_nodes,
                **({'pos': batch.pos} if self.with_3d and hasattr(batch, "pos") else {}),
                **({'z': batch.z} if self.with_3d and hasattr(batch, "z") else {}),
                # smiles = data.smiles
                #num_nodes_per_subgraph=data.num_nodes,
                # original_edge_index=data.edge_index, original_edge_attr=data.edge_attr)
        )

    def to_subgraphs(self, data):
        raise NotImplementedError


def parse_and_generate(data, group_counter, subgraph_type, generate_all, num_samples, with_3d=False):
    if subgraph_type.endswith('-ego'):
        # Extract the number before '-ego'
        n_str = subgraph_type.split('-ego')[0]
        try:
            n = int(n_str)
            return create_ego_pairs(data, group_counter, n, generate_all, num_samples, with_3d=with_3d)
        except ValueError:
            print(f"Warning: Could not parse ego network size from '{subgraph_type}'")
                
    elif subgraph_type == 'fragments':
        return create_fragment_pairs(data, group_counter, with_complement=True, num_samples=num_samples, with_3d=with_3d)
    
    elif subgraph_type == 'murcko':
        return create_murcko_pairs(data, group_counter, with_complement=True, num_samples=num_samples, with_3d=with_3d)
        
    elif subgraph_type == 'original':
        return create_full_pairs(data, group_counter, with_3d=with_3d)
    else:
        print(f"Warning: Unknown graph type '{subgraph_type}'")

def create_subgraph_config_name(subgraph_types, num_samples):
    name_parts = []
    for subgraph_type, num_sample in zip(subgraph_types, num_samples):
        # Format: {num_sample}k-{subgraph_type}
        part = f"{num_sample}k-{subgraph_type}"
        name_parts.append(part)
    
    subgraphs = "_".join(name_parts)
    return f"mix_{subgraphs}"

class SubgraphMix(Graph2Subgraph):
    def __init__(self, subgraph_types=[], num_samples=[], process_subgraphs=lambda x: x, pbar=None, with_3d=False):
        super().__init__(process_subgraphs, pbar, with_3d)
        self.generate_all = False
        self.subgraph_types = subgraph_types
        self.num_samples = num_samples
        
    def to_subgraphs(self, data):
        group_counter = 0
        subgraphs = []
        for idx, subgraph_type in enumerate(self.subgraph_types):
            subgraphsi, group_counter = parse_and_generate(data, group_counter, subgraph_type, self.generate_all, self.num_samples[idx], with_3d=self.with_3d)
            subgraphs += subgraphsi

        return subgraphs
    
class Original(Graph2Subgraph):
    def __init__(self, process_subgraphs=lambda x: x, pbar=None, with_3d=False):
        super().__init__(process_subgraphs, pbar, with_3d)
        
    def to_subgraphs(self, data):
        return [data]
    
def policy2transform(policy: str, subgraph_types=['2-ego'], num_samples = [2], process_subgraphs=lambda x: x, pbar=None, with_3d=False):
    assert len(subgraph_types) == len(num_samples), f"Please enter a number of sample for each subgraph_type in {subgraph_types} in {num_samples}"
    if policy == "subgraphs":
        return SubgraphMix(subgraph_types=subgraph_types, num_samples=num_samples, process_subgraphs=process_subgraphs, pbar=pbar, with_3d=with_3d)
    elif policy == "original":
        return Original(process_subgraphs=process_subgraphs, pbar=pbar, with_3d=with_3d)

    raise ValueError("Invalid subgraph policy type")