import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from tqdm import tqdm
import os

import numpy
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
torch.serialization.add_safe_globals([numpy.dtype])

import wandb as wandb
from preprocessing.dataloaders import get_data
from layers.ssl_models import get_ssl_model, get_probe_model
from preprocessing.utils.target_metric import get_target_metric
from utils.metrics import Evaluator, IsBetter
from utils.evaluation import run_probe_round
from utils.misc import args_canonize, args_unify, arg_parser, log_ssl_epoch, save_state, parse_name_cfg
from utils.training import Trainer, get_scheduler, get_optimizer, get_new_optimizer
# from postprocessing.tokenizer import Tokenizer 
import time

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

import torch.distributed as dist

import socket

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

import torch.distributed as dist

def setup():
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    if not dist.is_initialized():
        port = find_free_port()
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{port}',
            world_size=1,
            rank=0
        )
    
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

descriptors = {
    "drugs": ["energy", "ip", "ea", "chi"],
    "drugs-simple": ["energy", "ip", "ea", "chi"],
    "kraken": ["B5", "L", "burB5", "burL"],
    "kraken-simple": ["B5", "L", "burB5", "burL"],
    "BACE": ["bace_task"]
}

def main(args, wandb):
    wandb.config.update(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    create_data = args.create_data_only if hasattr(args, "create_data_only") else False
    save_ckpt = hasattr(args, "save_ckpt") and args.save_ckpt
    ssl = getattr(args, "ssl", True)
    with_3d = getattr(args.model, "with_3d", False)
    
    trn_loader, val_loader, _, task = get_data(args) 
    if create_data:
        print(f"Data processing done!")
        return

    setup()
    
    run_probe = getattr(args, "run_probe", False)
    if run_probe:    
        probe_models = []
        probe_trn_loaders, probe_val_loaders = [], []
        for dataset in args.probe.data.datasets:
            probe_trn_loader, probe_val_loader, _, _ = get_data(args, dataset=dataset, is_probe = True)
            probe_trn_loaders.append(probe_trn_loader)
            probe_val_loaders.append(probe_val_loader)
            probe_model =  get_probe_model(args, dataset, device)
            probe_models.append(probe_model)
        
        probe_target_metric, probe_criterion, probe_metric_name = get_target_metric(args.probe, ssl=False, dataset=dataset)
        probe_optimizer = get_optimizer(ssl=False, load_ckpt=False, model=probe_model, args=args.probe)
        probe_scheduler = get_scheduler(args.probe, target_metric=probe_target_metric, optimizer=probe_optimizer)
        probe_trainer = Trainer(
            task=task,
            criterion=probe_criterion,
            evaluator=Evaluator(probe_metric_name),
            target_metric=probe_target_metric,
            norm_target=getattr(args.probe, "norm_target", False),
            grad_clip_val=getattr(args.probe, "grad_clip_val", 1),
            device=device,
        )
        isBetter = IsBetter(probe_metric_name)
    
    target_metric, criterion, metric_name = get_target_metric(args, dataset=args.data.dataset)
    # target_metric, criterion, metric_name = get_target_metric(args)
    trainer = Trainer(
        task=task,
        criterion=criterion,
        evaluator=Evaluator(metric_name),
        target_metric=target_metric,
        norm_target=getattr(args, "norm_target", False),
        grad_clip_val=getattr(args, "grad_clip_val", 1),
        device=device,
    )
    

    global_epoch = 1
    for _run in range(args.num_runs):
        logging.info(f"Run {_run}")
        # args.tokenizer = Tokenizer(args.tokenizer_path) if hasattr(args, "tokenizer_path") else "tokenizer.json"

        model = get_ssl_model(args, device) 
        copy_model = get_ssl_model(args, device) 
        targets_encoder = copy_model.encoder #copy.deepcopy(model.encoder)
        targets_encoder.load_state_dict(model.encoder.state_dict())

        # print("Full Context Encoder:")
        # print(model)
        # print("Full Target Encoder")
        # print(targets_encoder)

        ipe = len(trn_loader.loader) # for momentum scheduler
        ipe_val = len(val_loader.loader) # for lr scheduler
        total_steps = int(ipe * (args.max_epoch + 1) * args.ipe_scale)
        momentum_scheduler = [
            args.ema[0] + i * (args.ema[1] - args.ema[0]) / total_steps # Precompute the schedule (we could use cosine instead if you want)
            for i in range(total_steps + 1)
        ]

        # add some noise to initial encoder
        # targets_encoder = copy.deepcopy(model.encoder)
        for param_k in targets_encoder.parameters():
            param_k.data += 0.001 * torch.randn_like(param_k.data)

        # ptimizer = get_optimizer(ssl=ssl, load_ckpt=False, model=model, args=args)
        optimizer = get_new_optimizer(model, args)
        scheduler = get_scheduler(
            args, target_metric=target_metric, optimizer=optimizer, ipe=ipe_val
        )
        trainer.clear_stats()

        best_val_loss = 1e5
        best_probe_metric = 1e5 # for now our metric for downstream is the average of all features
        trn_loss, val_loss = 1e5, 1e5
        epochs_no_improve = 0
        
        pbar = tqdm(range(1, args.max_epoch + 1))
        step = 0
        for epoch in pbar:

            epoch_start_time = time.time()

            log_dict = {}
            if epoch < 2:
                num_params = sum(p.numel() for p in model.parameters())
                print("Initialized (Context) Model with {} parameters".format(num_params))
                wandb.log({"num_params": num_params})
                if hasattr(args, "log_grads") and args.log_grads:
                    wandb.watch(
                        model,
                        log="all",
                    )
            
            trainer.epoch = epoch
            if run_probe:
                log_dict_probe, mlp_metric = run_probe_round(global_epoch, targets_encoder, probe_models, probe_trn_loaders, probe_val_loaders, probe_trainer, probe_optimizer, probe_scheduler, device, args.probe)
                if mlp_metric is not None:
                    better_probe_metric, _ = isBetter(mlp_metric, best_probe_metric)
                    if better_probe_metric:
                        best_probe_metric = mlp_metric
                        if save_ckpt:
                            save_state(epoch, targets_encoder, optimizer, criterion, trn_loss, val_loss, None, None, f"{args.final_ckpt_path}_best_avg_probe.pth")
                    
            trn_loss, _, step = trainer.train_ssl_epoch(
                model, targets_encoder, trn_loader, optimizer, momentum_scheduler, step, args.debug, with_3d
            )
            val_loss, _ = trainer.eval_ssl_epoch(
                model, targets_encoder, val_loader, scheduler, None, validation=True, with_3d=with_3d
            )

            epoch_duration = time.time() - epoch_start_time
            
            log_dict = log_ssl_epoch(trn_loss, val_loss, scheduler)
            log_dict["epoch_duration"] = epoch_duration

            if run_probe:
                log_dict.update(log_dict_probe)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if save_ckpt:
                    save_state(epoch, targets_encoder, optimizer, criterion, trn_loss, val_loss, None, None, f"{args.final_ckpt_path}_best_ssl_loss.pth") 
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print("Early stopping triggered!")
                    break
                
            pbar.set_postfix(log_dict)
            wandb.log(log_dict, step=global_epoch)
            global_epoch += 1

        if save_ckpt:
            save_state(epoch, targets_encoder, optimizer, criterion, trn_loss, val_loss, None, None, f"{args.final_ckpt_path}_final.pth") 
            
    wandb.finish()

def run_dataset_main(args, dataset = None, seed = None):
    args.data.dataset = dataset if dataset is not None else args.data.dataset
    args.seed = seed if seed is not None else args.seed

    # Track filename once it's determined
    if not hasattr(args, 'final_ckpt_path'):
        base_dir = 'checkpoints'
        base_name = f'{args.wandb.name}'
        filename = os.path.join(base_dir, base_name)

        # Avoid overwriting only the first time
        if os.path.exists(f"{filename}.pth"):
            counter = 1
            name_root, ext = os.path.splitext(base_name)
            while True:
                new_name = f"{name_root}({counter}){ext}"
                new_path = os.path.join(base_dir, new_name)
                if not os.path.exists(new_path):
                    filename = new_path
                    break
                counter += 1

        # Save this path for future use
        args.final_ckpt_path = filename
        print(f"Checkpoint will be saved to: {args.final_ckpt_path}")  # only printed once

    wandb_name = parse_name_cfg(args)
    wandb_name = (
        args.wandb.name + wandb_name if hasattr(args.wandb, "name") else None
    )  # None for sweeps

    wandb.init(
        project=args.wandb.project,
        name=wandb_name,
        mode="online" if args.wandb.use_wandb and not args.debug else "disabled",
        group=args.wandb.group,
        config=vars(args),
        entity=args.wandb.entity,
    )

    main(args, wandb)

if __name__ == "__main__":
    _, args = arg_parser(default_file="cfgs/pretrain-default.yaml")
    args = args_unify(args_canonize(args))
    run_dataset_main(args)  

