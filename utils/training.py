import math

import torch
import torch.optim as optim
from torch.optim import Optimizer, AdamW #, Muon
from muon import Muon, MuonWithAuxAdam
from torch.optim import Optimizer, AdamW #, Muon
from muon import Muon, MuonWithAuxAdam
from utils.metrics import Evaluator

from schedulefree import AdamWScheduleFree

from torch.profiler import profile, record_function, ProfilerActivity

SCHEDULER_MODE = {
    "acc": "max",
    "mae": "min",
    "mae_kraken": "min",
    "mse": "min",
    "rocauc": "max",
    "rmse": "min",
    "ap": "max",
    "mrr": "max",
    "mrr_self_filtered": "max",
    "f1_macro": "max",
    "bce": "min"


}
class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr

# class CosineWDSchedule(object):

#     def __init__(
#         self,
#         optimizer,
#         ref_wd,
#         T_max,
#         final_wd=0.
#     ):
#         self.optimizer = optimizer
#         self.ref_wd = ref_wd
#         self.final_wd = final_wd
#         self.T_max = T_max
#         self._step = 0.

#     def step(self):
#         self._step += 1
#         progress = self._step / self.T_max
#         new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

#         if self.final_wd <= self.ref_wd:
#             new_wd = max(self.final_wd, new_wd)
#         else:
#             new_wd = min(self.final_wd, new_wd)

#         for group in self.optimizer.param_groups:
#             if ('WD_exclude' not in group) or not group['WD_exclude']:
#                 group['weight_decay'] = new_wd
#         return new_wd
        
class NoScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        pass


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int = 5,
    num_training_steps: int = 250,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(args, target_metric, optimizer, ipe=1):
    if hasattr(args, "scheduler_type") and args.scheduler_type == "cos_with_warmup_new": # JEPA CosWarmup Scheduler
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=int(args.scheduler_warmup*ipe),
            start_lr=args.start_lr, 
            ref_lr=args.lr,
            final_lr=args.final_lr,
            T_max=int(args.ipe_scale*args.max_epoch*ipe)
        )
    elif hasattr(args, "scheduler_type") and args.scheduler_type == "cos_with_warmup": # Andreis scheduler
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=(
                        args.scheduler_warmup if hasattr(args, "scheduler_warmup") else 5
                    ),
                    num_training_steps=(
                        args.scheduler_patience if hasattr(args, "scheduler_patience") else 50
                    ),
                )
    elif hasattr(args, "scheduler_type") and args.scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=SCHEDULER_MODE[target_metric],
            factor=0.5,
            patience=(
                args.scheduler_patience if hasattr(args, "scheduler_patience") else 50
            ),
            min_lr=1.0e-12,
        )
    else:
        scheduler = NoScheduler(optimizer)

    return scheduler

class CombinedOptimizer:
    def __init__(self, optim1, optim2):
        self.optim1 = optim1
        self.optim2 = optim2

    def step(self):
        self.optim1.step()
        self.optim2.step()

    def zero_grad(self):
        self.optim1.zero_grad()
        self.optim2.zero_grad()

class SchedulerWrapper:
    def __init__(self, sched1, sched2):
        self.sched1 = sched1
        self.sched2 = sched2

    def step(self, metric):
        self.sched1.step(metric)
        self.sched2.step(metric)

def get_new_optimizer(model, args):
    param_2d = []
    param_rest = []
    for name, param in model.named_parameters():
        if param.dim() == 2:
            param_2d.append(param)
        else:
            param_rest.append(param)

    # optimizer_1d = torch.optim.AdamW(param_rest, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer_2d = Muon(
    #     param_2d,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    # )
    
    # optimizer = CombinedOptimizer(optimizer_1d, optimizer_2d)

    # hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
    # hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
    # nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
    param_groups = [
        dict(params=param_2d, use_muon=True,
            lr=args.lr, weight_decay=args.weight_decay),
        dict(params=param_rest, use_muon=False,
            lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay),
    ]
    optimizer = MuonWithAuxAdam(param_groups)

    # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer_1d,
    #     mode="min",
    #     patience=args.scheduler_patience,
    #     factor=args.scheduler_factor,
    # )

    # scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer_2d,
    #     mode="min",
    #     patience=args.scheduler_patience,
    #     factor=args.scheduler_factor,
    # )

    # scheduler = SchedulerWrapper(scheduler1, scheduler2)

    return optimizer #, scheduler

def get_optimizer(ssl, load_ckpt, model, args):
    weight_decay = getattr(args, "weight_decay", 0)
    tf_weight_decay = getattr(args, "tf_weight_decay", 0)
    if not ssl: #and load_ckpt and args.freeze_backbone:
        # optimizer = AdamW(
        #     model.parameters(),
        #     lr=args.lr,
        #     weight_decay=weight_decay,
        # )
        # optimizer = AdamWScheduleFree(
        #     model.parameters(),
        #     lr=args.lr,
        #     weight_decay=args.weight_decay,
        #     # warmup_steps=10000, 
        # )
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            # warmup_steps=10000, 
        )
    else:
        gin_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "model_2d" in name: 
                gin_params.append(param)
            else:
                other_params.append(param)

        optimizer = AdamW(
            [
                {"params": gin_params, "weight_decay": weight_decay},
                {"params": other_params, "weight_decay": tf_weight_decay},
            ],
            lr=args.lr,
        )
    return optimizer

class Trainer:
    def __init__(
        self,
        task,
        criterion,
        evaluator: Evaluator,
        target_metric,
        norm_target,
        grad_clip_val,
        device,
        target_i=None,
        target_j=None,
    ):
        super(Trainer, self).__init__()

        # only supports graph clf for now

        self.criterion = criterion
        self.evaluator = evaluator
        self.target_metric = target_metric
        self.norm_target = norm_target
        self.grad_clip_val = grad_clip_val
        self.device = device
        self.epoch_nr = 0
        self.I = target_i
        self.J = target_j
        self.clear_stats()

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.best_tst_metric = None
        self.patience = 0
    
    def train_ssl_epoch(self, context_encoder, targets_encoder, trn_loader, optimizer, momentum_scheduler, step, debug=False, with_3d=False):
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     on_trace_ready=lambda p: p.export_chrome_trace("trace-after-nopad.json"),
        #     record_shapes=True) as prof:
        if True:
            context_encoder.train()
            self.epoch_nr += 1

            train_losses = []
            for data in trn_loader.loader:
                # Step 1: Forward
                h1, h2 = data #mask_collator_fn(data, self.device, sample_ratio=sample_ratio)
                h1 = h1.to(torch.device(self.device), non_blocking = True)
                h2 = h2.to(torch.device(self.device), non_blocking = True)
                if with_3d:
                    h_context = context_encoder(h1.x, h1.batch, h1.subgraph_idx_batch, h1.edge_index, h1.edge_attr, h1.pos, h1.z, h1.z_batch)
                    with torch.no_grad(): # necessary?
                        h_targets = targets_encoder(h2.x, h2.batch, h2.subgraph_idx_batch, h2.edge_index, h2.edge_attr, h2.pos, h2.z, h2.z_batch) #, ssl = True)
                else:
                    h_context = context_encoder(h1.x, h1.batch, h1.subgraph_idx_batch, h1.edge_index, h1.edge_attr) #, h1.pos, h1.z, h1.z_batch)
                    with torch.no_grad(): # necessary?
                        h_targets = targets_encoder(h2.x, h2.batch, h2.subgraph_idx_batch, h2.edge_index, h2.edge_attr) #, h2.pos, h2.z, h2.z_batch) #, ssl = True)

                loss = self.criterion(h_context, h_targets)

                #  Step 2. Backward & step
                loss.backward()
                for name, param in context_encoder.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient in {name}")
                            break  # or raise an error
                optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = momentum_scheduler[step]
                with torch.no_grad():
                    for param_q, param_k in zip(context_encoder.parameters(), targets_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
                step +=1
                train_losses.append(loss.detach())

                if debug:
                    counter += 1
                    if counter > 3:
                        break

                # prof.step()
        
        # print(prof.key_averages().table(sort_by="cuda_time_total"))

        # After training loop
        trn_loss = torch.stack(train_losses).mean().item()
        loss_centr = 0
    
        return trn_loss + loss_centr, 0, step
    
    def eval_ssl_epoch(self, context_encoder, targets_encoder, loader, scheduler=None, wd_scheduler=None,  validation=False, with_3d=False):
        context_encoder.eval()

        contexts = []
        targetss = []
        with torch.no_grad():
            for data in loader.loader:
                h1, h2 = data #mask_collator_fn(data, self.device, sample_ratio=sample_ratio)
                h1 = h1.to(torch.device(self.device), non_blocking = True)
                h2 = h2.to(torch.device(self.device), non_blocking = True)

                # h_context = context_encoder(h1.x, h1.batch, h1.subgraph_idx_batch, h1.edge_index, h1.edge_attr) #, is_context=True)
                # h_targets = targets_encoder(h2.x, h2.batch, h2.subgraph_idx_batch, h2.edge_index, h2.edge_attr) #, ssl = True)
                if with_3d:
                    h_context = context_encoder(h1.x, h1.batch, h1.subgraph_idx_batch, h1.edge_index, h1.edge_attr, h1.pos, h1.z, h1.z_batch)
                    h_targets = targets_encoder(h2.x, h2.batch, h2.subgraph_idx_batch, h2.edge_index, h2.edge_attr, h2.pos, h2.z, h2.z_batch) #, ssl = True)
                else:
                    h_context = context_encoder(h1.x, h1.batch, h1.subgraph_idx_batch, h1.edge_index, h1.edge_attr)#, h1.pos, h1.z, h1.z_batch)
                    h_targets = targets_encoder(h2.x, h2.batch, h2.subgraph_idx_batch, h2.edge_index, h2.edge_attr) #, h2.pos, h2.z, h2.z_batch) #, ssl = True)

                contexts.append(h_context)
                targetss.append(h_targets)

                scheduler.step()

            contexts = torch.cat(contexts, dim=0)
            targetss = torch.cat(targetss, dim=0)
            
            tst_loss = self.criterion(contexts, targetss).item()
        
        # if scheduler is not None:
        #     if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         scheduler.step(tst_loss)
        #     else:
                

        return tst_loss, 0

    def train_epoch(self, model, trn_loader, optimizer, debug=False, with_subgraphs=False, only_head=False, std_mean=None, with_3d=False):
        I = self.I
        J = self.J

        model.train()
        if debug:
                counter = 0

        train_losses = []
        preds = []
        labels = []
        
        loader = trn_loader.loader if hasattr(trn_loader, "loader") else trn_loader
        # print(len(loader.dataset))
        for data in loader:

            if only_head:
                X, y = data
                X = X.to(torch.device(self.device))
                y = y.to(torch.device(self.device)) if I is None else y[:, I].to(torch.device(self.device))

                batch_idx = torch.tensor([i // 2 for i in range(X.shape[0] * X.shape[1])]).to(torch.device(self.device))
                optimizer.zero_grad()
                out = model(X, batch_idx if with_subgraphs else None, pooled=with_subgraphs)
                
                if out.shape[1] == 1:
                    out = out.squeeze(-1) 
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(-1) 

                std_mean = std_mean if std_mean is not None else (trn_loader.std[0] if trn_loader.std[0] is not None else None)
                # std_mean = std_mean if I is None else (std_mean[0][I], std_mean[1][I]) if std_mean[0] is not None else None
                if self.norm_target and std_mean is not None:
                    with torch.no_grad():
                        mean, std = std_mean[0].to(self.device), std_mean[1].to(
                            self.device
                        )
                        mean = mean if I is None else (mean[I] if mean.dim() == 1 else mean[:, I])
                        std = std if I is None else (std[I] if std.dim() == 1 else std[:, I])
                        y = (y - mean) / std

                mask = ~torch.isnan(y) & ~torch.isnan(out) # necessary for tasks with missing values like in Tox21  
                loss = self.criterion(out[mask], y[mask].float())

                loss.backward()
                # clip grad
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #self.grad_clip_val)
                optimizer.step()

                if self.norm_target and std_mean is not None:
                    with torch.no_grad():
                        y = y * std + mean
                        out = out * std + mean

                preds.append(out)
                labels.append(y)

                train_losses.append(loss.item())

            else: 
                data = data.to(torch.device(self.device))
                y = data.y if I is None else data.y[:, I:J].to(torch.device(self.device))
                # Reshape y
                y = y.view(-1, 1) if y.dim() == 1 else y

                # skip for BN during training
                if data.batch.shape[0] == 1:
                    continue
                if hasattr(data, "x_batch") and data.x_batch.shape[0] == 1:
                    continue

                optimizer.zero_grad()
                out = model(data.x, 
                            data.subgraph_batch if with_subgraphs else data.batch, 
                            data.subgraph_idx_batch if with_subgraphs else None, 
                            data.edge_index, data.edge_attr,
                            pos = data.pos if with_3d else None,
                            z = data.z if with_3d else None,
                            z_batch = data.z_batch if with_3d else None, 
                            pooled=with_subgraphs,
                            features_only = False)
                
                # if out.shape[1] == 1:
                #     out = out.squeeze(-1) 
                # if y.dim() > 1 and y.shape[1] == 1:
                #     y = y.squeeze(-1) 

                std_mean = std_mean if std_mean is not None else (trn_loader.std if trn_loader.std is not None else None)
                # std_mean = std_mean if I is None else (std_mean[0][I], std_mean[1][I]) if std_mean[0] is not None else None
                if self.norm_target and std_mean is not None:
                    with torch.no_grad():
                        mean, std = std_mean[0].to(self.device), std_mean[1].to(
                            self.device
                        )
                        mean = mean[I:J] if I is not None else mean # (mean[I] if mean.dim() == 1 else mean[:, I])
                        std = std[I:J] if I is not None else std #if I is None else (std[I] if std.dim() == 1 else std[:, I])
                        if mean.dim() > 1 and mean.shape[1] == 1:
                            mean = mean.squeeze(-1) 
                            std = std.squeeze(-1) 
                        y = (y - mean) / std

                # mask = ~torch.isnan(y)
                # loss = self.criterion(out[mask], y[mask]) # All targets
                loss = self.criterion(out, y) # All targets
                loss.backward()
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} got updated, grad norm: {param.grad.norm().item()}")
                #     else:
                #         print(f"{name} did NOT get a grad")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #self.grad_clip_val)
                optimizer.step()

                if self.norm_target and std_mean is not None:
                    with torch.no_grad():
                        y = y * std + mean
                        out = out * std + mean

                preds.append(out)
                labels.append(y)
                train_losses.append(loss.item())

            if debug:
                counter += 1
                if counter > 3:
                    break    

        trn_loss = sum(train_losses) / len(train_losses)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # mask_labels = ~torch.isnan(labels)

        # trn_metric = self.evaluator(labels[mask_labels].float(), preds[mask_labels])
        trn_metric = self.evaluator(labels.float(), preds)

        return trn_loss, trn_metric

    def eval_epoch(self, model, val_loader, scheduler=None, validation=False, with_subgraphs=False, only_head=False, std_mean=None, with_3d=False):
        model.eval()
        preds = []
        labels = []
        tst_metric = []

        I = self.I
        J = self.J

        with torch.no_grad():
            loader = val_loader.loader if hasattr(val_loader, "loader") else val_loader
            for data in loader:
                if only_head:
                    X, y = data
                    X = X.to(torch.device(self.device))
                    y = y.to(torch.device(self.device)) if I is None else y[:, I:J].to(torch.device(self.device))
                    y = y.view(-1, 1) if y.dim() == 1 else y

                    batch_idx = torch.tensor([i // 2 for i in range(X.shape[0] * X.shape[1])]).to(torch.device(self.device))
                    out = model(X, batch_idx if with_subgraphs else None, pooled=with_subgraphs)
                    if out.shape[1] == 1:
                        out = out.squeeze(-1) 
                    if y.dim() > 1 and y.shape[1] == 1:
                        y = y.squeeze(-1) 

                    std_mean = std_mean if std_mean is not None else (val_loader.std if val_loader.std is not None else None)
                    if self.norm_target and std_mean is not None:
                        mean, std = std_mean[0].to(self.device), std_mean[1].to(
                            self.device
                        )
                        mean = mean if I is None else (mean[I] if mean.dim() == 1 else mean[:, I])
                        std = std if I is None else (std[I] if std.dim() == 1 else std[:, I])
                        out = out * std + mean

                    preds.append(out)
                    labels.append(y)

                else:
                    data = data.to(torch.device(self.device))
                    y = data.y if I is None else data.y[:, I:J].to(torch.device(self.device))

                    y = y.view(-1, 1) if y.dim() == 1 else y

                    out = model(data.x, 
                                data.subgraph_batch if with_subgraphs else data.batch, 
                                data.subgraph_idx_batch if with_subgraphs else None, 
                                data.edge_index, data.edge_attr, 
                                pos = data.pos if with_3d else None,
                                z = data.z if with_3d else None,
                                z_batch = data.z_batch if with_3d else None, 
                                pooled=with_subgraphs,
                                features_only = False)
                    # if out.shape[1] == 1:
                    #     out = out.squeeze(-1) 
                    # if y.dim() > 1 and y.shape[1] == 1:
                    #     y = y.squeeze(-1) 

                    if val_loader.std is not None and self.norm_target:
                        mean, std = val_loader.std[0].to(self.device), val_loader.std[1].to(
                            self.device
                        )
                        mean = mean[I:J] if I is not None else mean #(mean[I] if mean.dim() == 1 else mean[:, I])
                        std = std[I:J] if I is not None else std #if I is None else (std[I] if std.dim() == 1 else std[:, I])
                        if mean.dim() > 1 and mean.shape[1] == 1:
                            mean = mean.squeeze(-1) 
                            std = std.squeeze(-1) 
                        out = out * std + mean

                    preds.append(out)
                    labels.append(y)

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            # if only_head:
            # necessary for tasks with missing values like in Tox21          
            mask = ~torch.isnan(labels)
            # print(mask.shape)
            # print(preds[mask].shape)
            # print(labels[mask].shape)
            # tst_loss = self.criterion(preds[mask], labels[mask].float()).item()
            tst_loss = self.criterion(preds, labels.float()).item()
            # else:
                # tst_loss = self.criterion(preds, labels).item()  # fix for rmse
            
            tst_metric = self.evaluator(labels, preds)
            tst_metric_first = next(iter(tst_metric.values()))

        if validation and scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(tst_metric_first)
            else:
                scheduler.step()

        return tst_loss, tst_metric

