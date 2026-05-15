import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

def mask_subgraphs_by_group_together(data):
    subgraph_batch = data.subgraph_batch
    subgraph_ids = torch.arange(data.subgraph_idx.shape[0], device=data.subgraph_idx.device)
    
    subgraph_group_id = data.subgraph_group_id_batch
    group_ids = subgraph_group_id.unique()

    # GPU-friendly filtering
    sampled_mask = (subgraph_group_id[..., None] == group_ids).any(dim=1)
    selected_indices = sampled_mask.nonzero(as_tuple=False).squeeze()

    # For each pair randomly put one graph in the context list and the other in target
    pairs = selected_indices.view(-1, 2)
    choices = torch.randint(0, 2, (pairs.size(0),), device=selected_indices.device)
    subgraph_context_ids = pairs[torch.arange(pairs.size(0), device=selected_indices.device), choices]
    subgraph_target_ids = pairs[torch.arange(pairs.size(0), device=selected_indices.device), 1 - choices]

    # vectorized comparison without broadcasting
    node_mask_context = torch.isin(subgraph_batch, subgraph_context_ids)
    node_mask_target = torch.isin(subgraph_batch, subgraph_target_ids)
    subgraph_mask_context = torch.isin(subgraph_ids, subgraph_context_ids)
    subgraph_mask_target = torch.isin(subgraph_ids, subgraph_target_ids)

    return node_mask_context, subgraph_mask_context, node_mask_target, subgraph_mask_target

def mask_subgraphs_by_group_separated(data):
    subgraph_batch = data.subgraph_batch
    subgraph_ids = torch.arange(data.subgraph_idx.shape[0], device=data.subgraph_idx.device)
    
    subgraph_group_id = data.subgraph_group_id_batch
    group_ids = subgraph_group_id.unique()

    # GPU-friendly filtering
    sampled_mask = (subgraph_group_id[..., None] == group_ids).any(dim=1)
    selected_indices = sampled_mask.nonzero(as_tuple=False).squeeze()

    # For each pair randomly put one graph in the context list and the other in target
    pairs = selected_indices.view(-1, 2)
    choices = torch.randint(0, 2, (pairs.size(0),), device=selected_indices.device)
    subgraph_context_ids = pairs[torch.arange(pairs.size(0), device=selected_indices.device), choices]
    subgraph_target_ids = pairs[torch.arange(pairs.size(0), device=selected_indices.device), 1 - choices]

    node_masks_context = []
    subgraph_masks_context = []
    node_masks_target = []
    subgraph_masks_target = []

    for ctx_id, tgt_id in zip(subgraph_context_ids, subgraph_target_ids):
        # Per context
        node_mask_ctx = torch.isin(subgraph_batch, ctx_id) 
        subgraph_mask_ctx = torch.isin(subgraph_ids, ctx_id) 
        node_masks_context.append(node_mask_ctx)
        subgraph_masks_context.append(subgraph_mask_ctx)

        # Per target
        node_mask_tgt = torch.isin(subgraph_batch, tgt_id)
        subgraph_mask_tgt = torch.isin(subgraph_ids, tgt_id)
        node_masks_target.append(node_mask_tgt)
        subgraph_masks_target.append(subgraph_mask_tgt)

    return node_masks_context, subgraph_masks_context, node_masks_target, subgraph_masks_target

def mask_collator_fn_together(data, mol_idx, with_3d=False):
    with_3d = with_3d and hasattr(data, "pos")
    node_mask_context, subgraph_ids_context, node_mask_target, subgraph_ids_target = mask_subgraphs_by_group_together(data)
    
    edge_index_context, edge_attr_context = subgraph(node_mask_context, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes = data.num_nodes * data.num_subgraphs)
    x_context = data.x[node_mask_context]
    if with_3d:
        if data.pos[0].shape[0] != data.x.shape[0]:
            print("yo")
        pos_context = [posi[node_mask_context, :] for posi in data.pos]
        z_context = data.z[node_mask_context] 
    _, batch_context = data.subgraph_batch[node_mask_context].unique(return_inverse=True)
    subgraph_idx_batch_context = data.subgraph_idx[subgraph_ids_context]

    context_data = Data(
        x=x_context,
        edge_index=edge_index_context,
        edge_attr=edge_attr_context,
        batch=batch_context,
        subgraph_idx=subgraph_idx_batch_context,
        mol_idx=torch.full_like(subgraph_idx_batch_context, fill_value=mol_idx),
        **({'pos': pos_context} if with_3d else {}),
        **({'z': z_context} if with_3d else {}),
    )

    edge_index_targets, edge_attr_targets = subgraph(node_mask_target, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes = data.num_nodes * data.num_subgraphs )
    x_target = data.x[node_mask_target]
    if with_3d:
        pos_target = [posi[node_mask_target, :]for posi in data.pos]
        z_target = data.z[node_mask_target] 
    _, batch_target = data.subgraph_batch[node_mask_target].unique(return_inverse=True)
    subgraph_idx_batch_target = data.subgraph_idx[subgraph_ids_target]

    target_data = Data(
        x=x_target,
        edge_index=edge_index_targets,
        edge_attr=edge_attr_targets,
        batch=batch_target,
        subgraph_idx=subgraph_idx_batch_target,
        mol_idx=torch.full_like(subgraph_idx_batch_target, fill_value=mol_idx),
        **({'pos': pos_target} if with_3d else {}),
        **({'z': z_target} if with_3d else {}),
    )

    return context_data, target_data

def mask_collator_fn_separated(data, mol_idx, with_3d=False):
    all_context_data = []
    all_target_data = []
    
    node_mask_context_list, subgraph_ids_context_list, node_mask_target_list, subgraph_ids_target_list = mask_subgraphs_by_group_separated(data)
    
    for node_mask_context, subgraph_ids_context in zip(node_mask_context_list, subgraph_ids_context_list):
        edge_index_context, edge_attr_context = subgraph(node_mask_context, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes = data.num_nodes * data.num_subgraphs)
        x_context = data.x[node_mask_context]
        _, batch_context = data.subgraph_batch[node_mask_context].unique(return_inverse=True)
        subgraph_idx_batch_context = data.subgraph_idx[subgraph_ids_context]
        context_data = Data(
            x=x_context,
            edge_index=edge_index_context,
            edge_attr=edge_attr_context,
            batch=batch_context,
            subgraph_idx=subgraph_idx_batch_context,
            mol_idx=torch.full_like(subgraph_idx_batch_context, fill_value=mol_idx)
        )
        all_context_data.append(context_data)

    for node_mask_target, subgraph_ids_target in zip(node_mask_target_list, subgraph_ids_target_list):
        edge_index_targets, edge_attr_targets = subgraph(node_mask_target, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes = data.num_nodes * data.num_subgraphs )
        x_target = data.x[node_mask_target]
        _, batch_target = data.subgraph_batch[node_mask_target].unique(return_inverse=True)
        subgraph_idx_batch_target = data.subgraph_idx[subgraph_ids_target]
        target_data = Data(
            x=x_target,
            edge_index=edge_index_targets,
            edge_attr=edge_attr_targets,
            batch=batch_target,
            subgraph_idx=subgraph_idx_batch_target,
            mol_idx=torch.full_like(subgraph_idx_batch_target, fill_value=mol_idx)
        )
        all_target_data.append(target_data)

    return all_context_data, all_target_data

def process_sugraph_separated(dataset, with_3d=False):
    processed_pairs = []
    for idx, data in enumerate(tqdm(dataset, desc="CCreating context/target pairs batches")):
        context_data_list, target_data_list = mask_collator_fn_separated(data, idx, with_3d=with_3d)
        for context_data, target_data in zip(context_data_list, target_data_list):
            processed_pairs.append((context_data, target_data))
    return processed_pairs

def process_sugraph_together(dataset, with_3d=False):
    processed_pairs = []
    for idx, data in enumerate(tqdm(dataset, desc="Creating context/target pairs batches")):
        context_data, target_data = mask_collator_fn_together(data, idx, with_3d=with_3d)
        processed_pairs.append((context_data, target_data))
    return processed_pairs


# Old method for MLM 
def mask_node_features(data, mask_ratio): #, mask_token):
    '''
    Randomly replaces node features with a constant value mask_val based on a given probability mask_ratio
    Each graph in the batch is processed independently
    '''
    x, x_batch = data.x, data.x_batch
    masked_x = x.clone()
    mask = torch.zeros(0, 1, device=x.device)

    # Mask nodes per graph
    unique_graphs = x_batch.unique() # Get graph indices in the batch
    for graph_id in unique_graphs:
        node_mask = (x_batch == graph_id)
        num_nodes = node_mask.sum().item()

        batch_mask = torch.rand(num_nodes, device=masked_x.device) < mask_ratio  # Each node is masked with prob mask_ratio
        batch_mask = batch_mask.unsqueeze(1)  # Shape: (num_nodes, 1)
        mask = torch.cat((mask, batch_mask), dim=0)

        #masked_x[node_mask] = torch.where(batch_mask, mask_token, masked_x[node_mask]) # Replace where mask = True with mask_val

    return mask