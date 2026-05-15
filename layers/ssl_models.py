
import torch
from torch.nn import Linear
from torch import nn

import torch_scatter
from torch_geometric.nn import aggr
from layers.attention import TfModule
from layers.ssl_encoders import Encoder3D, Encoder2D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISABLE = False
MODE = "default"

INPUT_DIM_SIMPLE = 2
EDGE_INPUT_DIM_SIMPLE = 2

INPUT_DIM_FULL = 9
EDGE_INPUT_DIM_FULL = 3

OUTPUT_DIM = 1

dataset_output_dim = {
    "zinc": 1,
    "zinc-subset": 1,
    "zinc-full": 1,
    "qm9": 1,
    "esol": 1,
    "freesolv": 1,
    "lipo": 1,
    "bace": 1,
    "bbbp": 1,
    "clintox": 2,
    "sider": 27,
    "toxcast": 617,
    "tox21": 12,
    "muv": 17,
    "hiv":1, 
    "kraken": 4,
    "kraken_b5": 1,
    "kraken_l": 1,
    "kraken_burb5": 1,
    "kraken_burl": 1,
    "drugs": 4,
    "drugs-kraken": 1,
    "drugs_energy": 1,
    "drugs_ip": 1,
    "drugs_ea": 1,
    "drugs_chi": 1,
    "proteins": 1,
    
}

def get_probe_model(args, dataset, device):
    downstream_dim = args.model.tf_hidden_dim
    output_dim = dataset_output_dim[dataset.lower()] if dataset.lower() in dataset_output_dim else OUTPUT_DIM
    model = PoolAndProject(downstream_dim, output_dim, args.probe.subgraph_pool, "mlp").to(device)

    return model

def get_ssl_model(args, device):
    model = SSLFragESAN(
        args=args
    ).to(device)

    return model

def get_sl_model(args, device):
    model = FragESANEncoder(
        input_dim=INPUT_DIM_SIMPLE if args.data.simple else INPUT_DIM_FULL,
        params=args.model,
        dataset=args.data.dataset,
        ssl=False,
        predictor_use_tf=False,
        simple=args.data.simple,
        head = args.model.head
    ).to(device)

    return model
class PoolAndProject(nn.Module):
    def __init__(self, downstream_dim, output_dim, subgraph_pool, head):
        super().__init__()
        self.subgraph_pool = subgraph_pool
        if self.subgraph_pool == "deepset":
            self.subgraph_aggr = aggr.DeepSetsAggregation(
                local_nn=nn.Sequential(
                    nn.Linear(downstream_dim, downstream_dim),
                    nn.GELU(),
                    nn.Linear(downstream_dim//2, downstream_dim),
                    nn.GELU(),

                ),
                global_nn=nn.Sequential(
                    nn.Linear(downstream_dim, downstream_dim),
                    nn.GELU(),
                    nn.Linear(downstream_dim//2, downstream_dim),
                    nn.GELU(),
                )
            )
        if head == "mlp":
            self.head = torch.nn.Sequential(
                torch.nn.Linear(downstream_dim, downstream_dim//2),
                torch.nn.GELU(),
                torch.nn.Linear(downstream_dim//2, downstream_dim//4),
                torch.nn.GELU(),
                torch.nn.Linear(downstream_dim//4, output_dim)
            )
        elif head == "lin":
            self.head = torch.nn.Sequential(
                torch.nn.Linear(downstream_dim, output_dim)
            )
        else:
            raise ValueError("Please specify the downstream head")


    def pool(self, h_subgraph, subgraph_idx_batch):
        # Subgraph Pooling
        if self.subgraph_pool == "mean":
            h_graph = torch_scatter.scatter(src=h_subgraph, index=subgraph_idx_batch, dim=0, reduce="mean")
        elif self.subgraph_pool == "max":
            h_graph = torch_scatter.scatter(src=h_subgraph, index=subgraph_idx_batch, dim=0, reduce="max")
        elif self.subgraph_pool == "sum":
            h_graph = torch_scatter.scatter(src=h_subgraph, index=subgraph_idx_batch, dim=0, reduce="sum")
        elif self.subgraph_pool == "deepset":
            h_graph = self.subgraph_aggr(h_subgraph, index=subgraph_idx_batch)
        else:
            h_graph = h_subgraph # no pooling
        return h_graph
    
    def forward(self, h_subgraph, subgraph_idx_batch, pooled):
        if pooled:
            if h_subgraph.dim() == 3:
                h_subgraph_flat = h_subgraph.view(h_subgraph.shape[0] * h_subgraph.shape[1], h_subgraph.shape[2])
                h_graph = self.pool(h_subgraph_flat, subgraph_idx_batch)
            else:
                h_graph = self.pool(h_subgraph, subgraph_idx_batch)
            out = self.head(h_graph)
        else:
            out = self.head(h_subgraph)
        
        return out


def node_pool(out, batch, mask, nodes_pool):
    if nodes_pool == "cls":
        h_subgraph = out[:, 0, :] # CLS Token
    else:
        out_flat = out.reshape(-1, out.size(-1)) 
        if mask is not None:
            # Remove pad tokens if we have any
            mask_flat = mask.reshape(-1) 
            out_nopad = out_flat[mask_flat]
        else: 
            out_nopad = out_flat

        if nodes_pool == "mean":
            h_subgraph = torch_scatter.scatter(src=out_nopad, index=batch, dim=0, reduce="mean")
        elif nodes_pool == "max":
            h_subgraph = torch_scatter.scatter(src=out_nopad, index=batch, dim=0, reduce="max")
        else:
            raise ValueError("Invalid pooling method for Nodes Pooling - If you're using cls set pad_max_len = True")

    return h_subgraph
        
    
class FragESANEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim=INPUT_DIM_SIMPLE,
        params=None,
        dataset=None,
        ssl=True,
        predictor_use_tf=True,
        simple=False,
        head="mlp"
    ):
        super(FragESANEncoder, self).__init__()
        
        # Parameters
        self.dataset = dataset
        self.hidden_dim = params.hidden_dim
        self.output_dim = dataset_output_dim[dataset.lower()] if dataset.lower() in dataset_output_dim else OUTPUT_DIM
        self.use_tf = params.use_tf
        self.nodes_pool = params.nodes_pool
        self.subgraph_pool = params.subgraph_pool
        self.ssl = ssl
        self.predictor_use_tf = predictor_use_tf
        self.with_2d = getattr(params, "with_2d", True)

        # Tf Parameters
        self.tf_hidden_dim = params.tf_hidden_dim

        # Initial Node & Edge Embedding layers
        # self.node_embed = Linear(input_dim, self.hidden_dim) 
        edge_input_dim = EDGE_INPUT_DIM_SIMPLE if simple else EDGE_INPUT_DIM_FULL
        if dataset.startswith("qm9"):
            self.preembed = Linear(11, 9)
        
        if dataset.startswith("zinc") or dataset.startswith("spice"):
            self.node_embed = ZINCAtomEncoder(self.hidden_dim)
            self.edge_encoder = ZINCBondEncoder(self.hidden_dim)
        elif dataset.startswith("qm9"):
            self.node_embed = Qm9AtomEncoder(self.hidden_dim)
            self.edge_encoder = QM9BondEncoder(self.hidden_dim)
        else:
            self.node_embed = Linear(input_dim, self.hidden_dim) 
            self.edge_encoder = torch.nn.Linear(edge_input_dim, self.hidden_dim)


        if params.with_3d:
            self.encoder = Encoder3D(self.edge_encoder, params, with_2d=self.with_2d)
        else:
            self.encoder = Encoder2D(self.edge_encoder, params)

        # Readout layers
        if not ssl:
            if self.use_tf:
                downstream_dim = self.tf_hidden_dim
            else:
                downstream_dim = self.hidden_dim
            self.head = head
            self.downstream = PoolAndProject(downstream_dim, self.output_dim, self.subgraph_pool, self.head)
            
    
    def forward(self, x, batch, subgraph_idx_batch, edge_index, edge_attr, pos=None, z=None, z_batch=None, with_node_pool = True, pooled=False, features_only=True):
        
        h = self.node_embed(x.float()) if x is not None else None
        
        out, mask = self.encoder(h, batch, edge_index, edge_attr, pos, z, z_batch) 
        if not with_node_pool:
            return out, mask
        
        # Nodes Pooling
        h_subgraph = node_pool(out, batch, mask, self.nodes_pool)
        
        return h_subgraph if features_only or self.ssl else self.downstream(h_subgraph, subgraph_idx_batch, pooled)
    
class FragESANPredictor(torch.nn.Module):
    def __init__(
        self,
        params,
        dataset,
        hidden_dim,
        n_proj_layers,
        bn_proj,
        use_tf = True,
    ):
        super(FragESANPredictor, self).__init__()
        
        self.use_tf = use_tf
        if use_tf:
            self.tf = TfModule(
                dim_h=params.tf_hidden_dim,
                input_dim_h=params.tf_hidden_dim,
                num_heads=params.n_tf_heads,
                num_layers=params.n_tf_layers,
                dropout=params.tf_dropout,
                new_arch=False,
            )
        
        self.nodes_pool = params.nodes_pool

        # Projector
        projector_layers = []
        for _ in range(n_proj_layers):
            projector_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if bn_proj:
                projector_layers.append(torch.nn.LayerNorm(hidden_dim))    
            projector_layers.append(torch.nn.GELU())

        self.projector_mlp = torch.nn.Sequential(*projector_layers)

    def forward(self, h_context, batch,mask):
        if self.use_tf:
            out = self.tf(h_context) if self.use_tf else h_context
            h_context = node_pool(out, batch, mask, self.nodes_pool) # Nodes Pooling
        h_context = self.projector_mlp(h_context)
        return h_context

class SSLFragESAN(torch.nn.Module):
    def __init__(
        self,
        args=None,
    ):
        super(SSLFragESAN, self).__init__()

        # Encoder
        input_dim = INPUT_DIM_SIMPLE if args.data.simple else INPUT_DIM_FULL
        self.encoder = FragESANEncoder(input_dim=input_dim,
                                       params=args.model, dataset=args.data.dataset, ssl=args.ssl, 
                                       predictor_use_tf=args.predictor.use_tf,
                                    #    predictor_hidden_dim=args.predictor.hidden_dim,
                                       simple=args.data.simple,
                                       head = args.model.head if not args.ssl else 'lin')

        self.hidden_dim = args.predictor.tf_hidden_dim
            
        self.predictor = FragESANPredictor(args.predictor, 
                                           args.data.dataset, 
                                           self.hidden_dim, 
                                           args.predictor.n_mlp_layers, 
                                           args.predictor.bn_proj)
        
    def forward(self, x, batch, subgraph_idx_batch, edge_index, edge_attr, pos=None, z=None, z_batch=None):
        h_context, mask = self.encoder(x, batch, subgraph_idx_batch, edge_index, edge_attr, pos, z, z_batch, with_node_pool = False)
        h_context = self.predictor(h_context, batch, mask)
        return h_context

class ZINCAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCAtomEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        return self.embedding(x)
    
class Qm9AtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(Qm9AtomEncoder, self).__init__()
        
        # Define the number of categories for each of the 11 features
        self.num_categories = [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21]  # example sizes
        
        # One embedding per feature
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(num_embeddings=n, embedding_dim=hidden)
            for n in self.num_categories
        ])
        
        for emb in self.embeddings:
            torch.nn.init.xavier_uniform_(emb.weight.data)
    
    def forward(self, x):
        x = x.long()  # Ensure x is of type long for embedding
        out = sum(self.embeddings[i](x[:, i]) for i in range(11))
        return out  # shape: (num_atoms, hidden)

class QM9BondEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(4, hidden)
    
    def forward(self, edge_attr):
        if edge_attr is not None:
            return self.linear(edge_attr.float())  # → (num_edges, hidden)
        return None
    
class ZINCBondEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCBondEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, edge_attr):
        if edge_attr is not None:
            return self.embedding(edge_attr.long())
        else:
            return None