from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_scatter import scatter
# from torchmetrics.functional import mean_squared_error


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class IsBetter:
    """
    A comparator for different metrics, to unify >= and <=

    """

    def __init__(self, task_type):
        self.task_type = task_type

    def __call__(self, val1: float, val2: Optional[float]) -> Tuple[bool, float]:
        if val2 is None:
            return True, val1

        if self.task_type.lower() in [
            "rmse",
            "mae",
            "mse",
            "mse_kraken",
            "mae_drugs",
            "mae_drugs_ip",
            "mae_drugs_ea",
            "mae_drugs_chi",
            "mae_kraken",
            "mae_kraken_l",
            "mae_kraken_b5",
            "mae_kraken_burb5",
            "mae_kraken_burl",
            "mae_drugs",
            "mse_drugs",
            "mae_spice",
            "mae_qm9",
            "zinc_mse",
            "zinc_mae"
        ]:
            better = val1 < val2
            the_better = val1 if better else val2
            return better, the_better
        elif self.task_type in [
            "rocauc",
            "acc",
            "f1_macro",
            "ap",
            "mrr",
            "mrr_self_filtered",
        ] or 'rocauc' in self.task_type.lower() or 'acc' in self.task_type.lower() :
            better = val1 >= val2
            the_better = val1 if better else val2
            return better, the_better
        else:
            raise ValueError


def pre_proc(y1: Union[torch.Tensor, np.ndarray], y2: Union[torch.Tensor, np.ndarray]):
    if len(y1.shape) == 1:
        y1 = y1[:, None]
    if len(y2.shape) == 1:
        y2 = y2[:, None]
    if isinstance(y1, torch.Tensor):
        y1 = y1.detach().cpu().numpy()
    if isinstance(y2, torch.Tensor):
        y2 = y2.detach().cpu().numpy()
    return y1, y2

from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score

def eval_rocauc(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str = "rocauc") -> Dict[str, float]:
    """
    Compute ROC-AUC averaged across tasks.
    Supports both single-label and multi-label cases.
    """
    rocauc_list = []

    # Ensure y_true and y_pred are 2D: [num_samples, num_tasks]
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    for i in range(y_true.shape[1]):
        # Only compute AUC if both positive and negative labels exist
        col_true = y_true[:, i].detach().cpu().numpy()
        col_pred = y_pred[:, i].detach().cpu().numpy()

        if np.any(col_true == 1) and np.any(col_true == 0):
            # Ignore NaNs (if any)
            is_labeled = ~np.isnan(col_true)
            if np.sum(is_labeled) > 0:  # sanity check
                rocauc_list.append(roc_auc_score(col_true[is_labeled], col_pred[is_labeled]))

        # # AUC is only defined when there is at least one positive data.
        # if np.any(y_true[:, i] == 1) and np.any(y_true[:, i] == 0):
        #     # ignore nan values
        #     is_labeled = y_true[:, i] == y_true[:, i]
        #     rocauc_list.append(
        #         roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
        #     )

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively and negatively labeled data available. Cannot compute ROC-AUC."
        )

    return {metric_name: sum(rocauc_list) / len(rocauc_list)}
    


def eval_acc(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str) -> Dict[str, float]:
    """
    eval accuracy (potentially multi task)

    :param y_true:
    :param y_pred:
    :return:
    """
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        correct = correct.cpu().detach().numpy()
        acc = float(np.sum(correct)) / len(correct)
        acc_list.append(acc)

    return {metric_name: sum(acc_list) / len(acc_list)}


def eval_rmse(y_true: torch.Tensor, y_pred: torch.Tensor, metric_name: str = "rmse") -> Dict[str, float]:
    mean_squared_error = torch.nn.MSELoss(reduction="mean")
    root_mean_squared_error = RMSELoss()
    rmse = root_mean_squared_error(y_pred, y_true)
    mse = mean_squared_error(y_pred, y_true)
    
    # Reduce dependencies
    # rmse = mean_squared_error(y_pred, y_true, squared=False)
    # mse = mean_squared_error(y_pred, y_true, squared=True)

    return {metric_name: rmse.item(), "mse": mse.item()}


def eval_rmse_(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str = "rmse") -> Dict[str, float]:
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(
            np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean())
        )

    return {metric_name: sum(rmse_list) / len(rmse_list)}


def eval_F1macro(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str = "f1_macro") -> Dict[str, float]:
    f1s = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        f1 = f1_score(y_true[is_labeled, i], y_pred[is_labeled, i], average="macro")
        f1s.append(f1)

    return {metric_name: sum(f1s) / len(f1s)}


def _eval_mrr_batch(y_pred_pos, y_pred_neg, pos_edge_batch_index):
    concat_pos_neg_pred = torch.cat([y_pred_pos, y_pred_neg], dim=1)
    argsort = torch.argsort(concat_pos_neg_pred, dim=1, descending=True)
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    mrr_list = scatter(
        1.0 / ranking_list.to(torch.float), pos_edge_batch_index, dim=0, reduce="mean"
    )
    return mrr_list


def eval_mrr_batch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    npreds: torch.Tensor,
    nnodes: torch.Tensor,
    edge_label_idx: torch.Tensor,
) -> Dict[str, float]:
    device = y_true.device
    # un-batch the edge label index
    num_graphs = len(nnodes)

    offset = torch.cat([nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]])
    offset = torch.repeat_interleave(offset, npreds)
    edge_label_idx = edge_label_idx - offset[None]

    arange_num_graphs = torch.arange(num_graphs, device=device)  # a shared tensor
    edge_batch_index = torch.repeat_interleave(arange_num_graphs, npreds)

    # get positive edges
    pos_edge_index = edge_label_idx[:, y_true == 1]
    num_pos_edges_list = scatter(y_true.long(), edge_batch_index, dim=0, reduce="sum")
    assert num_pos_edges_list.min() > 0
    num_pos_edges = num_pos_edges_list.sum()
    pos_edge_batch_index = edge_batch_index[y_true == 1]
    pred_pos = y_pred[
        pos_edge_batch_index, pos_edge_index[0], pos_edge_index[1]
    ].reshape(num_pos_edges, 1)

    # get negative edges
    # pad some out of range entries
    y_pred[
        arange_num_graphs.repeat_interleave(nnodes.max() - nnodes),
        :,
        torch.cat([torch.arange(n, nnodes.max(), device=device) for n in nnodes]),
    ] -= float("inf")

    neg_mask = torch.ones(num_pos_edges, nnodes.max(), dtype=torch.bool, device=device)
    neg_mask[torch.arange(num_pos_edges, device=device), pos_edge_index[1]] = False
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0], :][neg_mask].reshape(
        num_pos_edges, nnodes.max() - 1
    )
    mrr_list_raw = _eval_mrr_batch(pred_pos, pred_neg, pos_edge_batch_index)

    # filtered
    y_pred[pos_edge_batch_index, pos_edge_index[0], pos_edge_index[1]] -= float("inf")
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0], :]
    mrr_list_filtered = _eval_mrr_batch(pred_pos, pred_neg, pos_edge_batch_index)

    diag_arange = torch.arange(nnodes.max(), device=device)  # a shared tensor
    # self filtered
    y_pred[:, diag_arange, diag_arange] -= float("inf")
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0], :]
    mrr_list_self_filtered = _eval_mrr_batch(pred_pos, pred_neg, pos_edge_batch_index)

    return {
        "mrr_raw": mrr_list_raw.mean(),
        "mrr_filtered": mrr_list_filtered.mean(),
        "mrr_self_filtered": mrr_list_self_filtered.mean(),
    }


def eval_mse_kraken(y_true: torch.tensor, y_pred: torch.tensor, metric_name= "mse_kraken") -> Dict[str, float]:
    mse = torch.nn.MSELoss(reduction="none")
    losses = mse(y_pred, y_true)
    # ['sterimol_B5', 'sterimol_L', 'sterimol_burB5', 'sterimol_burL']
    return {
        "B5": losses[:, 0].mean().item(),
        "L": losses[:, 1].mean().item(),
        "burB5": losses[:, 2].mean().item(),
        "burL": losses[:, 3].mean().item(),
        "mse_kraken": losses.mean().item(),
    }


def eval_mae_kraken(y_true: torch.tensor, y_pred: torch.tensor, metric_name="mae_kraken") -> Dict[str, float]:
    mae = torch.nn.L1Loss(reduction='none')
    losses = mae(y_pred, y_true)
    return {
        "mae_kraken_b5": losses[:, 0].mean().item(),
        "mae_kraken_l": losses[:, 1].mean().item(),
        "mae_kraken_burb5": losses[:, 2].mean().item(),
        "mae_kraken_burl": losses[:, 3].mean().item(),
        "mae_kraken": losses.mean().item(),
    }

def eval_mae_drugs(y_true: torch.tensor, y_pred: torch.tensor, metric_name="mae_drugs") -> Dict[str, float]:
    mae = torch.nn.L1Loss(reduction='none')
    losses = mae(y_pred, y_true)
    return {
        "mae_drugs_ip": losses[:, 0].mean().item(),
        "mae_drugs_ea": losses[:, 1].mean().item(),
        "mae_drugs_chi": losses[:, 2].mean().item(),
        "mae_drugs": losses.mean().item(),
    }


def eval_mae(y_true: torch.tensor, y_pred: torch.tensor, metric_name: str) -> Dict[str, float]:
    mae = torch.nn.L1Loss()
    losses = mae(y_pred, y_true)
    return {metric_name: losses.mean().item()}

def eval_mse(y_true: torch.tensor, y_pred: torch.tensor, metric_name: str) -> Dict[str, float]:
    mse = torch.nn.MSELoss()
    losses = mse(y_pred, y_true)
    return {metric_name: losses.mean().item()}


class Evaluator:
    def __init__(self, task_type):
        self.task_type = task_type

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        npreds: torch.Tensor = None,
        nnodes: torch.Tensor = None,
        edge_label_idx: torch.Tensor = None,
    ):
        if 'rocauc' in self.task_type.lower():
            func = eval_rocauc
        elif 'rmse' in self.task_type.lower():
            func = eval_rmse
        elif 'mse_kraken' in self.task_type.lower(): 
            func = eval_mse_kraken
        elif '_mse' in self.task_type.lower(): 
            func = eval_mse
        elif 'mae_kraken' in self.task_type.lower():
            if 'mae_kraken_' in self.task_type.lower():
                func = eval_mae
            else:
                func = eval_mae_kraken
        elif 'mae_drugs' in self.task_type.lower():
            if 'mae_drugs_' in self.task_type.lower():
                func = eval_mae
            else:
                func = eval_mae_drugs
        elif 'acc' in self.task_type.lower():
            if y_pred.shape[1] == 1:
                # binary
                y_pred = (y_pred > 0.0).to(torch.int)
            else:
                if y_true.dim() == 1 or y_true.shape[1] == 1:
                    # multi class
                    y_pred = torch.argmax(y_pred, dim=1)
                else:
                    # multi label
                    raise NotImplementedError
            func = eval_acc
        elif 'f1_macro' in self.task_type.lower(): 
            assert y_pred.shape[1] > 1, "assumed not binary"
            y_pred = torch.argmax(y_pred, dim=1)
            func = eval_F1macro
        elif 'mae' in self.task_type.lower():
            func = eval_mae
        elif "mrr" in self.task_type:
            return eval_mrr_batch(y_true, y_pred, npreds, nnodes, edge_label_idx)
        else:
            raise ValueError(f"Unexpected task type {self.task_type}")

        # if self.task_type not in ["rmse", "mse_kraken", "mae_kraken", "mae"]:
        #     y_true, y_pred = pre_proc(y_true, y_pred)

        metric = func(y_true, y_pred, self.task_type)
        return metric
