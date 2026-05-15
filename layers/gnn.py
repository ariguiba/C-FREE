from typing import Union

import torch
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import GINEConv as PyGGINEConv
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.typing import OptPairTensor, OptTensor, Size

from torch_geometric.utils import dense_to_sparse, to_dense_adj
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LayerNorm
from torch import nn



class GINEConv(PyGGINEConv):

    def __init__(self, bond_encoder, mlp, **kwargs):
                #  in_channels, out_channels, **kwargs): #mlp

        # mlp = MLP(
        #     channel_list=[in_channels, out_channels, out_channels],
        #     act="gelu",
        #     dropout=0.1,
        # )
        super().__init__(nn=mlp, **kwargs)
        # self.bond_encoder = torch.nn.Sequential(
        #     bond_encoder, PyGLinear(-1, in_channels)
        # )
        self.bond_encoder = bond_encoder

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.bond_encoder is not None and edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr.float())

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        return x_j if edge_attr is None else x_j + edge_attr


def pad_subgraphs_new(h, batch, pad_token):
    device = h.device
    hidden_dim = h.size(-1)
    batch = batch.to(device)
    
    # Get number of graphs and number of nodes per graph (all tensors)
    num_graphs = batch.max() + 1
    node_counts = torch.bincount(batch, minlength=num_graphs)
    max_len = node_counts.max()
    
    # Create mask: True where position < node_count for that graph
    pos_matrix = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
    mask = pos_matrix < node_counts.unsqueeze(1)  # [num_graphs, max_len]
    
    # Initialize padded tensor
    padded_h = pad_token.expand(num_graphs, max_len, hidden_dim).clone()
    
    # Compute position of each node within its graph using vectorized operations
    # Create a tensor that counts occurrences of each graph ID up to each position
    batch_expanded = batch.unsqueeze(0)  # [1, num_nodes]
    batch_positions = torch.arange(len(batch), device=device).unsqueeze(1)  # [num_nodes, 1]
    
    # For each position i, count how many times batch[i] appears in batch[0:i]
    # This creates a mask where mask[i,j] = True if batch[i] == batch[j] and j < i
    same_graph_mask = (batch_expanded == batch.unsqueeze(1)) & (torch.arange(len(batch), device=device).unsqueeze(0) < batch_positions)
    pos_in_graph = same_graph_mask.sum(dim=1)  # Sum along the "previous positions" dimension
    
    # Use advanced indexing to scatter the features
    padded_h[batch, pos_in_graph] = h
    
    return padded_h, mask

# efficient padding
def pad_subgraphs(h, batch, pad_token):
    device = h.device
    hidden_dim = h.size(1)
    batch = batch.to(device)

    # Get number of graphs and number of nodes per graph
    num_graphs = batch.max().item() + 1
    node_counts = torch.bincount(batch, minlength=num_graphs)
    max_len = node_counts.max().item()

    # Allocate padded tensors
    padded_h = pad_token.expand(num_graphs, max_len, hidden_dim).clone()
    mask = torch.zeros(num_graphs, max_len, dtype=torch.bool, device=device)

    # Build per-graph position and scatter the features
    pos_in_graph = torch.cat([torch.arange(n, device=device) for n in node_counts], dim=0)

    # pos_in_graph = torch.arange(batch.size(0), device=device) - torch.cumsum(
    #     torch.bincount(batch, minlength=num_graphs)[batch], dim=0
    # ) + torch.bincount(batch, minlength=num_graphs)[batch] - 1
        
    padded_h[batch, pos_in_graph] = h
    mask[batch, pos_in_graph] = True # True is a real node, False if pad node

    return padded_h, mask

class SimpleGIN(torch.nn.Module):
    def __init__(self, 
                 hidden_dim,
                 n_gnn_layers,
                 bn_gin,
                 per_layer = False,
                 per_layer_aggr = "mean",
                 bond_encoder=None,
                 pad_token=None
                 ):
        super(SimpleGIN, self).__init__()

        # Controls
        self.hidden_dim = hidden_dim 
        self.per_layer = per_layer
        self.per_layer_aggr = per_layer_aggr
        self.pad_token = pad_token

        if per_layer_aggr == "project":
            self.project_aggr = Sequential(
                Linear(self.hidden_dim * n_gnn_layers, self.hidden_dim), 
                LayerNorm(self.hidden_dim),
            )
        
        # GIN Layers
        self.gin_convs = torch.nn.ModuleList()
        for _ in range(n_gnn_layers - 1):
            if bn_gin:
                mlp = Sequential(
                    Linear(self.hidden_dim, self.hidden_dim), 
                    LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                    LayerNorm(self.hidden_dim),
                    )
                #, eps=0., train_eps=False)
                self.gin_convs.append(GINEConv(bond_encoder, mlp))
                # self.gin_convs.append(GINEConv(bond_encoder, self.hidden_dim, self.hidden_dim))
                
            else:
                mlp = Sequential(
                    Linear(self.hidden_dim, self.hidden_dim), 
                    nn.GELU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                    )
                # self.gin_convs.append(GINEConv(Sequential(
                #     Linear(self.hidden_dim, self.hidden_dim), 
                #     nn.GELU(),
                #     Linear(self.hidden_dim, self.hidden_dim),
                # ), eps=0., train_eps=False))
                self.gin_convs.append(GINEConv(bond_encoder,mlp))
                # self.gin_convs.append(GINEConv(bond_encoder, self.hidden_dim, self.hidden_dim))
        
    def forward(self, h, batch, edge_index, h_edge_attr):
        assert self.pad_token is not None
        self.pad_token.to(h.device)

        if self.per_layer: 
            gnn_tf_outputs = []
            initial_node_features, mask = pad_subgraphs(h, batch, pad_token=self.pad_token)
            gnn_tf_outputs.append(initial_node_features)

        # Apply GIN layers
        for gin_conv in self.gin_convs:
            initial_input = h
            h = gin_conv(h, edge_index, h_edge_attr)
            h += initial_input
            h = F.gelu(h)  # As in Andreis code

            if self.per_layer:
                node_features, mask = pad_subgraphs(h, batch, pad_token=self.pad_token)
                gnn_tf_outputs.append(node_features)
        
        if self.per_layer:
            if self.per_layer_aggr == "mean":
                return torch.stack(gnn_tf_outputs, dim=0).mean(dim=0), mask 
            elif self.per_layer_aggr == "concat":
                return torch.concat(gnn_tf_outputs, dim=-2), mask
            elif self.per_layer_aggr == "project":
                x = torch.concat(gnn_tf_outputs, dim=-1)
                return self.project_aggr(x), mask
            else:
                raise ValueError("Invalid aggregation method for the intermediate GNN representations - use 'mean'")

        else:
            return h, None
