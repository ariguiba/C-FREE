import torch
import numpy as np
from utils.prop import MySGConv 
import torch_scatter
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

def get_features_and_target(encoder, trn_set, device, with_subgraphs = False, args = None):
    with_3d = args.data.with_3d
    X, y = [], []
    baseline = args.baseline.run_baseline if hasattr(args, "baseline") else False
    if baseline:
        use_prop = args.baseline.use_prop
        kdegree = args.baseline.kdegree
        out_channels = args.baseline.out_channels_prop
        sgconv = MySGConv(in_channels=-1, out_channels=out_channels, K=kdegree, cached=False, bias=False).to(device)
    with torch.no_grad():
        # Add tqdm progress bar
        for _, batch in enumerate(tqdm(trn_set)):
            batch = batch.to(device)
            
            if baseline:
                x = batch.x.to(float)
                edge_index = batch.edge_index
                x_prop = sgconv(x, edge_index) if use_prop else x # Raw features instead
                features = torch_scatter.scatter(src=x_prop, index=batch.x_batch, dim=0, reduce="mean")
            else:
                # run through encoder only
                features = encoder(batch.x, 
                                   batch.subgraph_batch if with_subgraphs else batch.batch, 
                                   batch.subgraph_idx_batch if with_subgraphs else None, 
                                   batch.edge_index, batch.edge_attr, 
                                   pos = batch.pos if with_3d else None,
                                   z = batch.z if with_3d else None,
                                   z_batch = batch.z_batch if with_3d else None,
                                   pooled=False,
                                   features_only = True)
                
            hi = torch.isfinite(features).all()
            if with_subgraphs:
                num_graphs = max(batch.subgraph_idx_batch) + 1
                subgraphs_per_graph = (batch.subgraph_idx_batch == 0).sum().item()
                sorted_idx = batch.subgraph_idx_batch.argsort()
                X_sorted = features[sorted_idx]
                X_grouped = X_sorted.view(num_graphs, subgraphs_per_graph, -1)  # [num_graphs, 2, N]
                X.append(X_grouped.cpu().numpy())
            else:
                X.append(features.cpu().numpy())

            y.append(batch.y.cpu().numpy())
    
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

def get_best_probe_metric(dataset, probe_trn_loader, probe_val_loader, pooled, probe_model, trainer, optimizer, scheduler, std_mean, args = None):
    maes = []
    n_epochs = args.n_epochs

    # Is this the best way to do it? Probe with fix seed MLPs each time?
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    best_metric = 1e5
    prog_bar = tqdm(range(n_epochs), desc=f"Running Probe on {dataset}", leave=False)
    for _ in prog_bar:
        _, _ = trainer.train_epoch(
            probe_model, probe_trn_loader, optimizer, False, with_subgraphs=pooled, only_head=True, std_mean=std_mean
        )
        
        _, val_metric = trainer.eval_epoch(
            probe_model, probe_val_loader, scheduler, validation=True, with_subgraphs=pooled, only_head=True, std_mean=std_mean
        )
        if best_metric  > next(iter(val_metric.values())):
            best_metric = next(iter(val_metric.values()))
    
        maes.append(best_metric)
    prog_bar.reset()
    
    return np.mean(maes)

def run_probe_round(global_epoch, model_encoder, probe_models, trn_loaders, val_loaders, trainer, optimizer, scheduler, device, args):
    run_every = args.run_every
    pooled = not (args.data.policy == 'original')
    datasets = args.data.datasets

    linprobepoch = (global_epoch -1) % run_every == 0
    log_dict = {}
    avg_metric = None

    # Linear probing
    if linprobepoch:    
        # Freeze model
        if model_encoder is not None:
            for param in model_encoder.parameters():
                param.requires_grad = False
            model_encoder.eval()

        results = []
        for idx, dataset in enumerate(datasets):
            probe_model = probe_models[idx]
            trn_loader = trn_loaders[idx]
            val_loader = val_loaders[idx]

            std_mean = trn_loader.std
            X, y = get_features_and_target(model_encoder, trn_loader.loader, device, pooled, args)
            X_test, y_test = get_features_and_target(model_encoder, val_loader.loader, device, pooled, args)

            features_trn_loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=args.batch_size, shuffle=True)
            features_val_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=args.batch_size, shuffle=True)
        
            result = get_best_probe_metric(dataset, features_trn_loader, features_val_loader, pooled, probe_model, trainer, optimizer, scheduler, std_mean, args = args)
            log_dict[f"probe/{dataset}"] = result
            results.append(result)
        
        # Unfreeze model
        if model_encoder is not None:
            for param in model_encoder.parameters():
                param.requires_grad = True
            model_encoder.train()

        avg_metric = np.mean(results)
        log_dict[f"probe/avg"] = avg_metric

    return log_dict, avg_metric # our MLP metric = average of all target 

