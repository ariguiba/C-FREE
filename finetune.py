import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb as wandb
import torch
import os
import numpy
import socket
from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
torch.serialization.add_safe_globals([numpy.dtype])

from preprocessing.dataloaders import get_data
from layers.ssl_models import get_sl_model
from preprocessing.utils.target_metric import get_target_metric
from utils.metrics import Evaluator, IsBetter
from utils.misc import args_canonize, args_unify, save_state, parse_name_cfg, log_epoch, arg_parser
from utils.training import Trainer, get_scheduler, get_optimizer, get_new_optimizer
# from postprocessing.tokenizer import Tokenizer
from utils.evaluation import get_features_and_target

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

def run_dataset_main(args, dataset = None, seed = None):
    args.data.dataset = dataset if dataset is not None else args.data.dataset
    args.seed = seed if seed is not None else args.seed

    # Determine filename to avoid overwritting existing checkpoints
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
        print(f"Checkpoint will be saved to: {args.final_ckpt_path}") 

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

def main(args, wandb):
    wandb.config.update(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssl = args.ssl if hasattr(args, "ssl") else False    
    save_ckpt = hasattr(args, "save_ckpt") and args.save_ckpt
    load_ckpt = args.load_ckpt if hasattr(args, "load_ckpt")  else False
    pool_subgraphs = not (args.data.policy == 'original')

    trn_loader, val_loader, tst_loader, task = get_data(args) 
    target_metric, criterion, metric_name = get_target_metric(args, dataset=args.data.dataset)
    trainer = Trainer(
        task=task,
        criterion=criterion,
        evaluator=Evaluator(metric_name),
        target_metric=target_metric,
        norm_target=getattr(args, "norm_target", False),
        grad_clip_val=getattr(args, "grad_clip_val", 1),
        device=device,
        target_i=getattr(args, "target_i", None),
        target_j=getattr(args, "target_j", None),
    )
    isBetter = IsBetter(metric_name)

    global_epoch = 1
    best_avg_tst = []
    for _run in range(args.num_runs):
        logging.info(f"Run {_run}")
        # args.tokenizer = Tokenizer(args.tokenizer_path) if hasattr(args, "tokenizer_path") else "tokenizer.json"
        model = get_sl_model(args, device) # we use target encoder for fine-tuning
        print("Downstream Encoder")
        print(model)

        if load_ckpt:
            checkpoint = torch.load(f"checkpoints/{args.backbone_file}.pth", weights_only=False)
            state_dict = checkpoint["state_dict"]
            new_state_dict = state_dict
            load_result = model.load_state_dict(new_state_dict, strict=False)
            print("Missing keys:", load_result.missing_keys)
            print("Unexpected keys:", load_result.unexpected_keys)

            if args.freeze_backbone: # train only MLP head
                for name, param in model.named_parameters():
                    if not name.startswith("downstream") and not name.startswith("subgraph_aggr"):
                        param.requires_grad = False

        if args.freeze_backbone:
            X_train, y_train = get_features_and_target(model, trn_loader.loader, device, with_subgraphs = pool_subgraphs, args = args)
            X_val, y_val = get_features_and_target(model, val_loader.loader, device, with_subgraphs = pool_subgraphs, args = args)
            X_tst, y_tst = get_features_and_target(model, tst_loader.loader, device, with_subgraphs = pool_subgraphs, args = args)

            cached_trn_loader = DataLoader(TensorDataset( torch.tensor(X_train), torch.tensor(y_train)), batch_size=args.batch_size, shuffle=True)
            cached_val_loader = DataLoader(TensorDataset( torch.tensor(X_val), torch.tensor(y_val)), batch_size=args.batch_size, shuffle=True)
            cached_tst_loader = DataLoader(TensorDataset( torch.tensor(X_tst), torch.tensor(y_tst)), batch_size=args.batch_size, shuffle=True)

        if args.use_muon:
            optimizer = get_new_optimizer(model, args)
        else:
            optimizer = get_optimizer(ssl, load_ckpt, model.downstream if args.freeze_backbone else model, args)

        scheduler = get_scheduler(args, target_metric=target_metric, optimizer=optimizer)
        
        trainer.clear_stats()

        best_val_loss = 1e5
        best_val_metric = {}
        best_tst_metric = {}
        epochs_no_improve = 0

        pbar = tqdm(range(1, args.max_epoch + 1))
        for epoch in pbar:
            if epoch < 2:
                num_params = sum(p.numel() for p in model.parameters())
                print("Initialized (Context) Model with {} parameters".format(num_params))
                logging.info(f"Number of parameters: {num_params}")
                wandb.log({"num_params": num_params})
            if hasattr(args, "log_grads") and args.log_grads:
                wandb.watch(
                    model,
                    log="all",
                )
            
            trainer.epoch = epoch
            std = trn_loader.std if trn_loader.std is not None else None
            trn_loss, val_loss, _, val_metric, tst_metric = train(model.downstream if args.freeze_backbone else model, pool_subgraphs,
                    trainer, optimizer, scheduler,
                    cached_trn_loader if args.freeze_backbone else trn_loader,
                    cached_val_loader if args.freeze_backbone else val_loader, 
                    cached_tst_loader if args.freeze_backbone else tst_loader,
                    std,
                    args)
            
            log_dict = log_epoch(trn_loss, val_loss, scheduler, epochs_no_improve)
            
            if epoch < 2:
                # update the metrics 
                for key in val_metric.keys():
                    # initialize if not seen before
                    if key not in best_val_metric:
                        best_val_metric[key] = val_metric[key]
                        if args.eval_test:
                            best_tst_metric[key] = tst_metric[key]
            if epoch > 1:    
                curr_val_metric = next(iter(val_metric.values()))
                curr_best_val_metric = next(iter(best_val_metric.values()))
                better_val, the_better = isBetter(curr_val_metric, curr_best_val_metric)
                if better_val:
                # if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    for key in val_metric.keys():
                        better_val, the_better = isBetter(val_metric[key], best_val_metric[key])
                        # better_loss, the_better = isBetter(tst_metric[key], best_tst_metric[key])
                        if better_val: #or better_loss:
                            epochs_no_improve = 0
                            best_val_metric[key] = val_metric[key]
                            if args.eval_test:
                                best_tst_metric[key] = tst_metric[key]
                    if save_ckpt:
                            save_state(epoch, model, optimizer, criterion, 0, val_loss, val_metric, tst_metric, f"{args.final_ckpt_path}_best_loss.pth") 
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= args.patience:
                        print("Early stopping triggered!")
                        break
                
            for name, value in val_metric.items():
                log_dict[f"val/{name}"] = value
            if tst_metric:
                for name, value in tst_metric.items():
                    log_dict[f"tst/{name}"] = value
                
            pbar.set_postfix(log_dict)
            wandb.log(log_dict, step=global_epoch)
            global_epoch += 1
        
        wandb.log({
            "best_val/": best_val_metric,
            "best_tst/": best_tst_metric,
        }, step=5000)

        if args.eval_test:
            tst_metric_first = next(iter(best_tst_metric.values()))
            best_avg_tst.append(tst_metric_first)

        if save_ckpt:
            save_state(epoch, model, optimizer, criterion, 0, val_loss, best_val_metric, best_tst_metric, f"{args.final_ckpt_path}_final.pth") 

    # wandb.log({
    #         "best_avg_tst/": numpy.mean(best_avg_tst),
    #         "best_std_tst/": numpy.std(best_avg_tst),
    #     }, step=5000)
    wandb.finish()


def train(model, pool_subgraphs, trainer: Trainer, optimizer, scheduler, 
          trn_loader, val_loader, tst_loader, std, args): 
    if args.use_muon:
        setup()
    trn_loss, _ = trainer.train_epoch(
        model, trn_loader, optimizer, args.debug, with_subgraphs=pool_subgraphs, only_head=args.freeze_backbone, std_mean=std, with_3d=args.model.with_3d
    )

    val_loss, val_metric = trainer.eval_epoch(
        model, val_loader, scheduler, validation=True, with_subgraphs=pool_subgraphs, only_head=args.freeze_backbone, std_mean=std, with_3d=args.model.with_3d
    )

    if args.eval_test:
        tst_loss, tst_metric = trainer.eval_epoch(
        model, tst_loader, scheduler, validation=False, with_subgraphs=pool_subgraphs, only_head=args.freeze_backbone, std_mean=std, with_3d=args.model.with_3d
    )   
            
    return trn_loss, val_loss, tst_loss if args.eval_test else _, val_metric, tst_metric if args.eval_test else []

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

if __name__ == "__main__":
    _, args = arg_parser(default_file="cfgs/finetune-default.yaml")
    args = args_unify(args_canonize(args))
    run_dataset_main(args)  

