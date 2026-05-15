from rdkit.Chem import Draw
import torch
from rdkit import Chem
import torch
from types import SimpleNamespace
import numpy as np

from datasets.datasets import get_data
from layers.ssl_models import SimpleGIN
from utils.evaluation import run_linear_probe
import wandb

from datasets.datasets import get_data
from layers.ssl_models import get_sl_model

class Args:
        def __init__(self):
            self.dataset = "proteins"
            self.data_path = "./datasets"
            self.n_conformers = 1
            self.batch_size = 16
            self.debug = False
            self.use_both_datasets = False
            self.model = SimpleNamespace(
                hidden_dim = 128,
                n_gnn_layers = 4,
                # Tf params
                tf_hidden_dim = 256,
                n_tf_heads = 4,
                n_tf_layers = 4,
                tf_dropout = 0.05,

                # Controls
                loss = "CosSim", 
                bn_gin = False,
                use_tf = True,
                nodes_pool = "cls", 
                subgraph_pool = "max", # Options: attaggr or mean or max
                n_proj_layers = 2,
                bn_proj = False,
                per_layer = True,
                per_layer_aggr = "project",
                new_arch = False)
            self.policy = 'ego_nets'
            self.ssl = True
            self.target_k = 0
            self.train_subset_perc = 1
            self.max_epoch = 10

def load_model(args, device):
    model = get_sl_model(args, device)
    file = "best-models/drugs-kraken-B1.3"
    checkpoint = torch.load(f"checkpoints/{file}.pth", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model

if __name__ == "__main__":

    args = Args()
    device = "cuda"
    model = load_model(args, device)
    trn_loader, val_loader, _, task = get_data(args)

    X, y = [], []
    num_batches = 10000000
    with torch.no_grad():
        for i, batch in enumerate(trn_loader.loader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            
            no_mask_nodes = torch.ones(batch.subgraph_batch.shape[0], dtype=torch.bool)
            no_mask_subgraph = torch.ones(batch.subgraph_idx_batch.shape[0], dtype=torch.bool)
            features = model(batch, no_mask_nodes, no_mask_subgraph, batch.edge_index, batch.edge_attr, with_readout=False)

            X.append(features.cpu().numpy())
            y.append(batch.y.cpu().numpy())
    
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    
    
    
    


