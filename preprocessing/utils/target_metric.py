import torch

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
    
def get_target_metric(args, ssl=None, dataset=None):
    dataset = args.dataset.lower() if dataset is None else dataset.lower()
    ssl = args.ssl if ssl is None else ssl

    if ssl:
        target_metric = "mse"
        metric_name = "GEOM_MSE"
        criterion = torch.nn.MSELoss()
        
    elif dataset == "zinc":
        # target_metric = "mse"
        # metric_name = "ZINC_MSE"
        # criterion = torch.nn.MSELoss(reduction="mean")
        target_metric = "mae"
        metric_name = "ZINC_mae"
        criterion = torch.nn.L1Loss()
    
    elif dataset == "chiro":
        target_metric = "acc"
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        metric_name = "chiro_acc"

    elif dataset in ["esol", "freesolv", "lipo"]:
        target_metric = "rmse"
        metric_name = f"{dataset}_RMSE"
        criterion = RMSELoss()

    elif dataset in ["bace", "bbbp", "hiv", "clintox", "toxcast", "tox21", "muv", "sider"]:
        target_metric = "rocauc"
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        metric_name = f"{dataset}_ROCAUC"
        
    elif dataset.startswith("kraken"):
        criterion = torch.nn.L1Loss()
        if '_b5' in dataset:
            target_metric = "mae"
            metric_name = "mae_kraken_b5"
        elif '_l' in dataset:
            target_metric = "mae"
            metric_name = "mae_kraken_l"
        elif '_burb5' in dataset:
            target_metric = "mae"
            metric_name = "mae_kraken_burb5"
        elif '_burl' in dataset:
            target_metric = "mae"
            metric_name = "mae_kraken_burl"
        else:
            target_metric = "mae_kraken"
            metric_name = "mae_kraken"

    elif dataset.startswith("drugs"):
        criterion = torch.nn.L1Loss()
        if '_ip' in dataset:
            target_metric = "mae"
            metric_name = "mae_drugs_ip"
        elif '_ea' in dataset:
            target_metric = "mae"
            metric_name = "mae_drugs_ea"
        elif '_chi' in dataset:
            target_metric = "mae"
            metric_name = "mae_drugs_chi"
        else:
            target_metric = "mae_drugs"
            metric_name = "mae_drugs"
    elif dataset.startswith("exp"):
        target_metric = "acc"
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        metric_name = "exp_acc"

    elif dataset.startswith("spice"):
        target_metric = "mae"
        criterion = torch.nn.L1Loss()
        metric_name = "mae_spice"

    elif dataset.startswith("qm9"):
        target_metric = "mae"
        criterion = torch.nn.L1Loss()
        metric_name = "mae_qm9"
        
    else:
        raise NotImplementedError(f"Dataset {args.data.dataset} not implemented")

    return target_metric, criterion, metric_name