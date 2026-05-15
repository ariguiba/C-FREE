import random
import numpy as np
 
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, k_hop_subgraph, subgraph

from utils.fragmentation.fragmentations import RingsPaths

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected

def create_murcko_pairs(data, group_counter, with_complement=False, num_samples=0, with_3d=False):
    # Convert graph to SMILES if available, or construct from data
    if hasattr(data, 'smiles') and data.smiles:
        mol = Chem.MolFromSmiles(data.smiles)
    else:
        # If SMILES not available, we need to construct molecule from graph
        # This requires more complex logic depending on your data format
        raise ValueError("SMILES string required for Murcko scaffold generation")
    
    if mol is None:
        raise ValueError("Invalid molecule")
    
    # Generate Murcko scaffold
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
    except:
        # If scaffold generation fails, return original molecule duplicated
        subgraphs = []
        subgraphs.append(
            Data(
                x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                subgraph_idx=torch.tensor(0), subgraph_group_id=torch.tensor(group_counter),
                **({'pos': data.pos} if with_3d and hasattr(data, 'pos') else {}),
                **({'z': data.z} if with_3d and hasattr(data, 'z') else {}),
            )
        )
        subgraphs.append(
            Data(
                x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                subgraph_idx=torch.tensor(1), subgraph_group_id=torch.tensor(group_counter),
                **({'pos': data.pos} if with_3d and hasattr(data, 'pos') else {}),
                **({'z': data.z} if with_3d and hasattr(data, 'z') else {}),
            )
        )
        return subgraphs, group_counter
    
    # Map scaffold atoms to original molecule atoms
    scaffold_match = mol.GetSubstructMatch(scaffold)
    
    if len(scaffold_match) == 0:
        # No scaffold found, return original molecule duplicated
        subgraphs = []
        subgraphs.append(
            Data(
                x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                subgraph_idx=torch.tensor(0), subgraph_group_id=torch.tensor(group_counter),
                **({'pos': data.pos} if with_3d and hasattr(data, 'pos') else {}),
                **({'z': data.z} if with_3d and hasattr(data, 'z') else {}),
            )
        )
        subgraphs.append(
            Data(
                x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                subgraph_idx=torch.tensor(1), subgraph_group_id=torch.tensor(group_counter),
                **({'pos': data.pos} if with_3d and hasattr(data, 'pos') else {}),
                **({'z': data.z} if with_3d and hasattr(data, 'z') else {}),
            )
        )
        return subgraphs, group_counter
    
    # Create tensor of scaffold node indices
    scaffold_nodes = torch.tensor(list(scaffold_match), dtype=torch.long)

    # Extract scaffold subgraph
    mask = torch.tensor([
        (src.item() in scaffold_nodes and tgt.item() in scaffold_nodes) 
        for src, tgt in zip(data.edge_index[0], data.edge_index[1])
    ])
    
    scaffold_edge_index_raw = data.edge_index[:, mask]
    scaffold_edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None
    
    subset = torch.unique(scaffold_edge_index_raw)

    # Use subgraph() to get remapped edge_index
    scaffold_edge_index, _ = subgraph(
        subset= subset,
        edge_index=scaffold_edge_index_raw,
        relabel_nodes=True
    )

    if data.edge_attr is not None:
        scaffold_edge_index, scaffold_edge_attr = to_undirected(scaffold_edge_index, scaffold_edge_attr,
                                                                num_nodes=data.x.shape[0])
    else:
        scaffold_edge_index = to_undirected(scaffold_edge_index, scaffold_edge_attr,
                                            num_nodes=data.x.shape[0])
    
    
    # Get remapped edge_index for scaffold
    # scaffold_edge_index, scaffold_edge_attr = subgraph(
    #     subset=scaffold_nodes,
    #     edge_index=data.edge_index,
    #     edge_attr=data.edge_attr,
    #     relabel_nodes=True,
    #     num_nodes=data.num_nodes
    # )
    
    # if data.edge_attr is not None:
    #     scaffold_edge_index, scaffold_edge_attr = to_undirected(
    #         scaffold_edge_index, scaffold_edge_attr, num_nodes=data.x.shape[0]
    #     )
    # else:
    #     scaffold_edge_index = to_undirected(
    #         scaffold_edge_index, num_nodes=data.x.shape[0]
    #     )
    
    subgraphs = []
    subgraph_counter = 0
    
    if with_complement:
        # Create complement subgraph (non-scaffold atoms)
        all_nodes = torch.arange(data.num_nodes, device=scaffold_nodes.device)
        complement_nodes = all_nodes[~torch.isin(all_nodes, scaffold_nodes)]
        
        if complement_nodes.nelement() != 0:
            complement_edge_index, complement_edge_attr = subgraph(
                subset=complement_nodes,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )
            
            # Prepare 3D data if needed
            scaffold_pos = None
            complement_pos = None
            scaffold_z = None
            complement_z = None
            
            if with_3d and hasattr(data, 'pos'):
                if isinstance(data.pos, list):
                    scaffold_pos = [posi[scaffold_nodes, :] for posi in data.pos]
                    complement_pos = [posi[complement_nodes, :] for posi in data.pos]
                else:
                    scaffold_pos = data.pos[scaffold_nodes, :]
                    complement_pos = data.pos[complement_nodes, :]
                
                if hasattr(data, 'z'):
                    scaffold_z = data.z[scaffold_nodes]
                    complement_z = data.z[complement_nodes]
            
            # Add scaffold subgraph
            subgraphs.append(
                Data(
                    x=data.x[scaffold_nodes],
                    edge_index=scaffold_edge_index,
                    edge_attr=scaffold_edge_attr,
                    subgraph_idx=torch.tensor(subgraph_counter),
                    subgraph_group_id=torch.tensor(group_counter),
                    **({'pos': scaffold_pos} if scaffold_pos is not None else {}),
                    **({'z': scaffold_z} if scaffold_z is not None else {}),
                )
            )
            subgraph_counter += 1
            
            # Add complement subgraph
            subgraphs.append(
                Data(
                    x=data.x[complement_nodes],
                    edge_index=complement_edge_index,
                    edge_attr=complement_edge_attr,
                    subgraph_idx=torch.tensor(subgraph_counter),
                    subgraph_group_id=torch.tensor(group_counter),
                    **({'pos': complement_pos} if complement_pos is not None else {}),
                    **({'z': complement_z} if complement_z is not None else {}),
                )
            )
            subgraph_counter += 1
            group_counter += 1
        else:
            # Only scaffold exists (no side chains)
            subgraphs.append(
                Data(
                    x=data.x[scaffold_nodes],
                    edge_index=scaffold_edge_index,
                    edge_attr=scaffold_edge_attr,
                    subgraph_idx=torch.tensor(subgraph_counter),
                    subgraph_group_id=torch.tensor(group_counter),
                    **({'pos': data.pos[scaffold_nodes]} if with_3d and hasattr(data, 'pos') else {}),
                    **({'z': data.z[scaffold_nodes]} if with_3d and hasattr(data, 'z') else {}),
                )
            )
            subgraphs.append(
                Data(
                    x=data.x[scaffold_nodes],
                    edge_index=scaffold_edge_index,
                    edge_attr=scaffold_edge_attr,
                    subgraph_idx=torch.tensor(1),
                    subgraph_group_id=torch.tensor(group_counter),
                    **({'pos': data.pos[scaffold_nodes]} if with_3d and hasattr(data, 'pos') else {}),
                    **({'z': data.z[scaffold_nodes]} if with_3d and hasattr(data, 'z') else {}),
                )
            )
    else:
        # Only add scaffold subgraph
        scaffold_pos = None
        scaffold_z = None
        
        if with_3d and hasattr(data, 'pos'):
            if isinstance(data.pos, list):
                scaffold_pos = [posi[scaffold_nodes, :] for posi in data.pos]
            else:
                scaffold_pos = data.pos[scaffold_nodes, :]
            
            if hasattr(data, 'z'):
                scaffold_z = data.z[scaffold_nodes]
        
        subgraphs.append(
            Data(
                x=data.x[scaffold_nodes],
                edge_index=scaffold_edge_index,
                edge_attr=scaffold_edge_attr,
                subgraph_idx=torch.tensor(subgraph_counter),
                subgraph_group_id=torch.tensor(group_counter),
                **({'pos': scaffold_pos} if scaffold_pos is not None else {}),
                **({'z': scaffold_z} if scaffold_z is not None else {}),
            )
        )
    
    return subgraphs, group_counter


def create_fragment_pairs(data, group_counter, with_complement = False, num_samples=0, with_3d=False):
    rp = RingsPaths()
    graph = rp(data)

    frag_id_to_type = dict(
        [frag_info for frag_infos in graph.substructures for frag_info in frag_infos if frag_info])
    max_frag_id = max([frag_id for frag_infos in graph.substructures for (
        frag_id, _) in frag_infos], default=-1)
    frag_representation = torch.zeros(max_frag_id + 1, rp.vocab_size)
    frag_representation[list(range(max_frag_id + 1)), [frag_id_to_type[frag_id]
                                                        for frag_id in range(max_frag_id + 1)]] = 1
    graph.fragments = frag_representation
    edges = [[node_id, frag_id] for node_id, frag_infos in enumerate(
        graph.substructures) for (frag_id, _) in frag_infos]

    # same list but sorted fragment-wise instead of node wise
    edges_np = np.array(edges)
    sorted_idx = edges_np[:, 1].argsort()
    sorted_edges = edges_np[sorted_idx]

    frags_list = torch.tensor(sorted_edges[:, 1], dtype=torch.long)
    node_list = torch.tensor(sorted_edges[:, 0], dtype=torch.long) 
    
    subgraphs = []
    subgraph_counter = 0
    added = 0 
    rand_val = 1 if num_samples == 0 else 0.5 # if num_samples = 0 we add all elements, else we sample #num_samples elements
    for batch_idx in range(max_frag_id + 1):
        indices = (frags_list == batch_idx).nonzero(as_tuple=True)[0]
        selected_nodes = node_list[indices]

        mask = torch.tensor([(src in selected_nodes and tgt in selected_nodes) for src, tgt in zip(graph.edge_index[0], graph.edge_index[1])])
        subgraph_edge_index_raw = graph.edge_index[:, mask]
        subgraph_edge_attr = graph.edge_attr[mask]
        subset = torch.unique(subgraph_edge_index_raw)

        # Use subgraph() to get remapped edge_index
        subgraph_edge_index, _ = subgraph(
            subset= subset,
            edge_index=subgraph_edge_index_raw,
            relabel_nodes=True
        )

        if data.edge_attr is not None:
            subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                    num_nodes=graph.x.shape[0])
        else:
            subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                num_nodes=graph.x.shape[0])
        
        
        # Randomly decide if element is added or not 
        rng = random.Random()  # creates a new random generator instance
        rng.seed() # seeds it from system entropy (same as no argument)
        if rng.random() <= rand_val:
            if with_complement:
                all_nodes = torch.arange(data.num_nodes, device=subset.device)
                subset_comp = all_nodes[~torch.isin(all_nodes, subset)]
                subgraph_x_comp = data.x[subset_comp] if data.x is not None else None
                subgraph_edge_index_comp, subgraph_edge_attr_comp = subgraph(
                    subset_comp, data.edge_index, data.edge_attr,
                    relabel_nodes=True, num_nodes=data.num_nodes
                )

                if hasattr(data, "pos") and with_3d:
                    subgraph_pos = [posi[selected_nodes, :] for posi in data.pos] 
                    subgraph_pos_comp = [posi[subset_comp, :] for posi in data.pos] 
                    subgraph_z = data.z[selected_nodes]
                    subgraph_z_comp = data.z[subset_comp]

                if subset.nelement() != 0 and subset_comp.nelement() != 0:
                    subgraphs.append(
                        Data(
                            x=graph.x[selected_nodes], edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                            subgraph_idx=torch.tensor(subgraph_counter), subgraph_group_id=torch.tensor(group_counter),
                            **({'pos': subgraph_pos} if with_3d else {}),
                            **({'z': subgraph_z} if with_3d else {}),
                            # smiles=data.smiles
                        )
                    )
                    subgraph_counter += 1

                    subgraphs.append(
                        Data(
                            x=subgraph_x_comp, edge_index=subgraph_edge_index_comp, edge_attr=subgraph_edge_attr_comp,
                            subgraph_idx=torch.tensor(subgraph_counter), subgraph_group_id=torch.tensor(group_counter),
                            **({'pos': subgraph_pos_comp} if with_3d else {}),
                            **({'z': subgraph_z_comp} if with_3d else {}),
                            # smiles=data.smiles
                        )
                    )
                    group_counter += 1
                    subgraph_counter += 1
            
            else:
                if subset.nelement() != 0:
                    subgraphs.append(
                        Data(
                            x=graph.x[selected_nodes], edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                            subgraph_idx=torch.tensor(subgraph_counter), subgraph_group_id=torch.tensor(group_counter),
                            **({'pos': subgraph_pos} if with_3d else {}),
                            **({'z': subgraph_z} if with_3d else {}),
                            # smiles=data.smiles
                        )
                    )
                    subgraph_counter += 1

            added += 1 
            if added == num_samples:
                break
    
    if len(subgraphs) == 0:
        subgraphs.append(
            Data(
            x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
            subgraph_idx=torch.tensor(0), subgraph_group_id=torch.tensor(group_counter),
            **({'pos': data.pos} if with_3d else {}),
            **({'z': data.z} if with_3d else {}),
            )
        )
        subgraphs.append(
            Data(
            x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
            subgraph_idx=torch.tensor(1), subgraph_group_id=torch.tensor(group_counter),
            **({'pos': data.pos} if with_3d else {}),
            **({'z': data.z} if with_3d else {}),
            )
        )

    return subgraphs, group_counter

def generate_ego_net_for_node(i, data, num_hops, subgraphs, subgraph_counter, group_counter, with_3d=False):
    subset, subgraph_edge_index, inv, edge_mask = k_hop_subgraph(
        i, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
    subgraph_x = data.x[subset] if data.x is not None else None
    if hasattr(data, "pos") and with_3d:
        subgraph_pos = [posi[subset, :] for posi in data.pos] 
        subgraph_z = data.z[subset]

    all_nodes = torch.arange(data.num_nodes, device=subset.device)
    subset_comp = all_nodes[~torch.isin(all_nodes, subset)]
    subgraph_x_comp = data.x[subset_comp] if data.x is not None else None
    if hasattr(data, "pos") and with_3d:
        subgraph_pos_comp = [posi[subset_comp, :] for posi in data.pos] 
        subgraph_z_comp = data.z[subset_comp]

    subgraph_edge_index_comp, subgraph_edge_attr_comp = subgraph(
        subset_comp, data.edge_index, data.edge_attr,
        relabel_nodes=True, num_nodes=data.num_nodes
    )
    
    if subset.nelement() == 0 or subset_comp.nelement() == 0:
        return subgraph_counter, group_counter
        
    subgraphs.append(
        Data(
            x=subgraph_x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
            subgraph_idx=torch.tensor(subgraph_counter), subgraph_group_id=torch.tensor(group_counter),
            **({'pos': subgraph_pos} if with_3d else {}),
            **({'z': subgraph_z} if with_3d else {}),
            # smiles=data.smiles
        )
    )
    subgraph_counter += 1

    subgraphs.append(
        Data(
            x=subgraph_x_comp, edge_index=subgraph_edge_index_comp, edge_attr=subgraph_edge_attr_comp,
            subgraph_idx=torch.tensor(subgraph_counter), subgraph_group_id=torch.tensor(group_counter),
            **({'pos': subgraph_pos_comp} if with_3d else {}),
            **({'z': subgraph_z_comp} if with_3d else {}),
            # smiles=data.smiles
        )
    )
    subgraph_counter += 1
    group_counter += 1

    return subgraph_counter, group_counter

def create_ego_pairs(data, group_counter, num_hops, generate_all = True, k = 1, with_3d=False):
    subgraphs = []
    subgraph_counter = 0
    if generate_all:
        for i in range(data.num_nodes):
            subgraph_counter, group_counter = generate_ego_net_for_node(
                i, data, num_hops, subgraphs, subgraph_counter, group_counter,with_3d=with_3d
            )
    else:
        g = torch.Generator(device=data.x.device)
        g.seed()  # Seeds with current time or system entropy
        random_nodes = torch.randint(0, data.num_nodes, (k,), generator=g, device=data.x.device)
        for i in random_nodes.tolist():  # convert to list to safely iterate
            subgraph_counter, group_counter = generate_ego_net_for_node(
                i, data, num_hops, subgraphs, subgraph_counter, group_counter,with_3d=with_3d
            )

    if subgraphs == []:
        subgraphs.append(
            Data(
            x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
            subgraph_idx=torch.tensor(0), subgraph_group_id=torch.tensor(group_counter),
            **({'pos': data.pos} if with_3d else {}),
            **({'z': data.z} if with_3d else {}),
            # smiles=data.smiles
            )
        )
        subgraphs.append(
            Data(
            x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
            subgraph_idx=torch.tensor(1), subgraph_group_id=torch.tensor(group_counter),
            **({'pos': data.pos} if with_3d else {}),
            **({'z': data.z} if with_3d else {}),
            # smiles=data.smiles
            )
        )
    return subgraphs, group_counter

def create_full_pairs(data, group_counter, with_3d=False):
    subgraphs = []
    subgraphs.append(
        Data(
        x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
        subgraph_idx=torch.tensor(0), subgraph_group_id=torch.tensor(group_counter),
        **({'pos': data.pos} if with_3d else {}),
        **({'z': data.z} if with_3d else {}),
        # smiles=data.smiles
        )
    )
    subgraphs.append(
        Data(
        x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
        subgraph_idx=torch.tensor(0), subgraph_group_id=torch.tensor(group_counter),
        **({'pos': data.pos} if with_3d else {}),
        **({'z': data.z} if with_3d else {}),
        # smiles=data.smiles
        )
    )
    return subgraphs, group_counter



# Fragments at ESAN style (with all nodes in the subgraphs)
def fragments_to_subgraphs(data):
    
    rp = RingsPaths()
    graph = rp(data)

    # fr = FragmentRepresentation()
    # graph = fr(graph)

    subgraphs = []

    # NEW FUNCTION
    frag_id_to_type = dict(
        [frag_info for frag_infos in graph.substructures for frag_info in frag_infos if frag_info])
    max_frag_id = max([frag_id for frag_infos in graph.substructures for (
        frag_id, _) in frag_infos], default=-1)
    frag_representation = torch.zeros(max_frag_id + 1, rp.vocab_size)
    frag_representation[list(range(max_frag_id + 1)), [frag_id_to_type[frag_id]
                                                        for frag_id in range(max_frag_id + 1)]] = 1
    graph.fragments = frag_representation
    edges = [[node_id, frag_id] for node_id, frag_infos in enumerate(
        graph.substructures) for (frag_id, _) in frag_infos]

    # same list but sorted fragment-wise instead of node wise
    edges_np = np.array(edges)
    sorted_idx = edges_np[:, 1].argsort()
    sorted_edges = edges_np[sorted_idx]

    frags_list = torch.tensor(sorted_edges[:, 1], dtype=torch.long)
    node_list = torch.tensor(sorted_edges[:, 0], dtype=torch.long) 

    for batch_idx in range(max_frag_id + 1):
        indices = (frags_list == batch_idx).nonzero(as_tuple=True)[0]
        selected_nodes = node_list[indices]

        mask = torch.tensor([(src in selected_nodes and tgt in selected_nodes) for src, tgt in zip(graph.edge_index[0], graph.edge_index[1])])
        subgraph_edge_index = graph.edge_index[:, mask]
        subgraph_edge_attr = graph.edge_attr[mask]

        if data.edge_attr is not None:
            subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                    num_nodes=graph.x.shape[0])
        else:
            subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                num_nodes=graph.x.shape[0])
            
        subgraphs.append(
            Data(
                x=graph.x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                subgraph_idx=torch.tensor(batch_idx), 
                # subgraph_node_idx=torch.arange(graph.x.shape[0]), #selected_nodes,
                num_nodes=graph.x.shape[0], #selected_nodes.shape[0] #
            )
        )
    
    if len(subgraphs) == 0:
        subgraphs = [
            Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
                    subgraph_idx=torch.tensor(0), subgraph_node_idx=torch.arange(graph.x.shape[0]),
                    num_nodes=graph.x.shape[0],
                    )
        ]
        # subgraphs = [
        #         Data(x=data.x_frag, edge_index=data.edge_index_frag, edge_attr=data.edge_attr_frag,
        #              subgraph_idx=data.subgraph_idx, subgraph_node_idx=data.node_frag, #torch.arange(data.num_nodes),
        #              num_nodes=data.num_nodes,
        #              )
        #     ]

    return subgraphs  

def create_ego_nets_only(data, num_hops):
    subgraphs = []

    for i in range(data.num_nodes):

        subset, _, _, edge_mask = k_hop_subgraph(i, num_hops, data.edge_index, relabel_nodes=False,
                                            num_nodes=data.num_nodes)
        subgraph_edge_index = data.edge_index[:, edge_mask]
        subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else data.edge_attr

        x = data.x
        subgraphs.append(
            Data(
                x=x[subset], edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                subgraph_idx=torch.tensor(i), 
                #subgraph_node_idx=torch.arange(data.num_nodes),
                #num_nodes=data.num_nodes,
            )
        )
    return subgraphs