import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import TfModule
from layers.gnn import SimpleGIN

from layers.model_utils import cutoff
from layers.schnet import MySchNet
from layers.painn import PaiNN
from layers.gnn import SimpleGIN
from layers.model_utils import radial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder2D(torch.nn.Module):
    def __init__(
        self,
        edge_encoder,
        params
        
    ):
        super(Encoder2D, self).__init__()

        # Parameters
        self.hidden_dim = params.hidden_dim
        self.use_tf = params.use_tf
        self.nodes_pool = params.nodes_pool

        # GNN Params
        self.per_layer = params.per_layer
        self.per_layer_aggr = params.per_layer_aggr
        self.n_gnn_layers = params.n_gnn_layers
        self.pad_token = nn.Parameter(torch.empty(1, self.hidden_dim))

        # Tf Parameters
        self.tf_hidden_dim = params.tf_hidden_dim
        self.n_tf_heads = params.n_tf_heads
        self.tf_dropout = params.tf_dropout
        self.n_tf_layers = params.n_tf_layers

        # GNN Encoder
        self.model_2d = SimpleGIN(self.hidden_dim, self.n_gnn_layers, params.bn_gin,
                                    per_layer = self.per_layer, per_layer_aggr = self.per_layer_aggr,
                                    bond_encoder=edge_encoder, pad_token=self.pad_token)
        # Tf Encoder
        if self.use_tf:
            if self.per_layer_aggr == "concat":
                tf_hidden_dim_input = self.hidden_dim * self.n_gnn_layers
                self.tf_hidden_dim = tf_hidden_dim_input

                if self.nodes_pool == "cls":
                    self.cls_token = nn.Parameter(torch.empty(1, tf_hidden_dim_input))

            else:
                tf_hidden_dim_input = self.hidden_dim
                if self.nodes_pool == "cls":
                    self.cls_token = nn.Parameter(torch.empty(1, self.hidden_dim))

            self.tf = TfModule(
                dim_h=self.tf_hidden_dim,
                input_dim_h=tf_hidden_dim_input,
                num_heads=self.n_tf_heads,
                num_layers=self.n_tf_layers,
                dropout=self.tf_dropout,
                new_arch=False,
            )
        
        self.init_tokens()

    def init_tokens(self):
        nn.init.normal_(self.pad_token, mean=0.0, std=0.1)
        if self.nodes_pool == "cls" and self.use_tf: # Use 'mean' pooling when not using tf
            nn.init.normal_(self.cls_token, mean=0.0, std=0.1)
        
    def forward(self, h, batch, edge_index, edge_attr, pos=None, z=None, z_batch=None):
        # we learn only node features, edge_attr is used only in the GNN to improve the node representations, but we don't add it to the tf.
        if not self.use_tf:
            assert self.nodes_pool != "cls", "CLS pooling is not supported when not using tf. Please change the pooling method to 'mean' or 'max' in the config file."

        # Apply GNN
        h_node, mask = self.model_2d(h, batch, edge_index, edge_attr)

        if self.use_tf:
            # Append CLS
            tokens = torch.cat(
                [self.cls_token.expand(h_node.shape[0], -1, h_node.shape[2]), h_node], dim=1
            ) if self.nodes_pool == "cls" else h_node

            # Apply Tf Encoder
            out = self.tf(tokens) if self.use_tf else h_node
        else:
            out = h_node

        return out, mask


class Encoder3D(torch.nn.Module):
    def __init__(
        self,
        edge_encoder,
        params,
        with_2d = True
        
    ):
        super(Encoder3D, self).__init__()

        # Parameters
        self.hidden_dim = params.hidden_dim
        self.use_tf = params.use_tf
        self.nodes_pool = params.nodes_pool

        # GNN Params
        self.per_layer = params.per_layer
        self.per_layer_aggr = params.per_layer_aggr
        self.n_gnn_layers = params.n_gnn_layers

        # Tf Parameters
        self.tf_hidden_dim = params.tf_hidden_dim
        self.n_tf_heads = params.n_tf_heads
        self.tf_dropout = params.tf_dropout
        self.n_tf_layers = params.n_tf_layers

        # Special tokens
        self.sep_token = nn.Parameter(torch.empty(1, self.hidden_dim))
        self.pad_token = nn.Parameter(torch.empty(1, self.hidden_dim))

        # GNN Encoder
        self.with_2d = with_2d
        if with_2d:
            self.model_2d = SimpleGIN(self.hidden_dim, self.n_gnn_layers, params.bn_gin,
                                        per_layer = self.per_layer, per_layer_aggr = self.per_layer_aggr,
                                        bond_encoder=edge_encoder, pad_token=self.pad_token)
                
        self.model_3d = Model3D(params)

        # Tf Encoder
        if self.use_tf:
            if self.per_layer_aggr == "concat":
                self.tf_hidden_dim = self.tf_hidden_dim

                if self.nodes_pool == "cls":
                    self.cls_token = nn.Parameter(torch.empty(1, self.hidden_dim))

            else:
                tf_hidden_dim_input = self.hidden_dim
                if self.nodes_pool == "cls":
                    self.cls_token = nn.Parameter(torch.empty(1, self.hidden_dim))

            self.tf = TfModule(
                dim_h=self.tf_hidden_dim,
                input_dim_h=self.hidden_dim, 
                num_heads=self.n_tf_heads,
                num_layers=self.n_tf_layers,
                dropout=self.tf_dropout,
                new_arch=False,
            )

        # Modality positional embeddings
        # We append one position embedding for each modality (NOT each molecule for each modality)
        self.pos_emb_2d = nn.Parameter(torch.empty(1, self.hidden_dim))
        self.pos_emb_3d = nn.Parameter(torch.empty(1, self.hidden_dim))
        

        self.init_tokens()

    def init_tokens(self):
        nn.init.normal_(self.pad_token, mean=0.0, std=0.1)
        nn.init.normal_(self.sep_token, mean=0.0, std=0.1)
        nn.init.normal_(self.pos_emb_3d, mean=0.0, std=0.1)
        nn.init.normal_(self.pos_emb_2d, mean=0.0, std=0.1)
        if self.use_tf and self.nodes_pool == "cls": # Use 'mean' pooling when not using tf
            nn.init.normal_(self.cls_token, mean=0.0, std=0.1)
        
    def forward(self, h, batch, edge_index, edge_attr, pos, z, z_batch):
        # we learn only node features, edge_attr is used only in the GNN to improve the node representations, but we don't add it to the tf.
        mask = None

        if not self.use_tf:
            assert self.nodes_pool != "cls", "CLS pooling is not supported when not using tf. Please change the pooling method to 'mean' or 'max' in the config file."

        if self.with_2d:
            # Apply GNN
            tokens_2d_no_sep, mask = self.model_2d(h, batch, edge_index, edge_attr)
            tokens_2d_no_sep += self.pos_emb_2d
            tokens_2d = tokens_2d_no_sep

        # If we use the tf or the 2d model, we add padding tokens to align the representations
        tokens_3d_no_sep = self.model_3d(h, pos, z, batch, train=True, pad_token=self.pad_token, pad=self.use_tf or self.with_2d)

        # Probably better way to do this without wasting time passing through GIN but we need the mask for padding
        if self.with_2d:
            tokens_3d = torch.cat(
                [self.sep_token.expand(tokens_3d_no_sep.shape[0], -1, -1), tokens_3d_no_sep],
                dim=1,
            )
            tokens = torch.cat([tokens_2d, tokens_3d], dim=1) 
        else:
            tokens_3d_no_sep += self.pos_emb_3d
            tokens = tokens_3d_no_sep

        if self.use_tf:
            # Append CLS
            tokens = torch.cat(
                [self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1
            )

            out = self.tf(tokens)
        else:
            out = tokens

        return out, mask

def pad_to_max_len(feats, max_len, pad_token):
    # pad to max_len with self.pad_token
    padded_tensor = torch.cat(
        [
            feats,
            pad_token.expand(
                max_len - feats.shape[0], -1
            ),
        ],
        dim=0,
    )
    return padded_tensor

class Model3D(nn.Module):
    def __init__(self, args):
        super(Model3D, self).__init__()

        model_3d_type = getattr(args, "model_3d_type", "SchNet")

        if model_3d_type == "SchNet":
            self.model_3d = MySchNet(
                hidden_channels=args.hidden_dim, num_filters=args.hidden_dim,
                args=args
            )
        elif model_3d_type == "PaiNN":
            n_interactions = getattr(args, "num_layers_3d", 3)
            n_rbf = getattr(args, "num_radial_3d", 20)
            cutoff_3d = getattr(args, "cutoff_3d", 5.0)
            self.model_3d = PaiNN(
                n_atom_basis=args.hidden_dim,
                n_interactions=n_interactions,
                radial_basis=radial.GaussianRBF(
                    n_rbf=n_rbf,
                    cutoff=cutoff_3d,
                ),
                cutoff_fn=cutoff.CosineCutoff(cutoff_3d),
            )
        
        pretrained = getattr(args, "pretrained_3d", None)
        if pretrained is not None:
            print(f"Loading pretrained 3D model from '{pretrained}'")
            state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)["state_dict"]
            self.model_3d.load_state_dict(state_dict)


    def forward(self, h, pos, z, z_batch, train=True, pad_token=None, pad=True):
        assert pad_token is not None

        model_3d_tf_outputs = []
        # check if pos is an array
        if not isinstance(pos, (list, tuple)):
            pos = [pos]
        for i in range(len(pos)):
            x_3d = self.model_3d(h, z, pos[i], z_batch)

            if pad:
                max_idx = z_batch.max()
                max_len = 0
                for j in torch.arange(max_idx + 1):
                    max_len = max(max_len, x_3d[z_batch == j].shape[0])

                node_features_list = [
                    pad_to_max_len(x_3d[z_batch == j], max_len=max_len, pad_token=pad_token)
                    for j in range(max_idx + 1)
                ]

                node_features = torch.stack(node_features_list)

                model_3d_tf_outputs.append(node_features)
            else:
                model_3d_tf_outputs.append(x_3d)

        #maybe also add avg here
        centroids_3d = torch.concat(model_3d_tf_outputs, dim=1)

        return centroids_3d
