import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from tqdm import tqdm
import numpy as np
import os

import numpy
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
torch.serialization.add_safe_globals([numpy.dtype])

import wandb as wandb
from datasets.datasets import get_data
from utils.misc import Config, args_canonize, args_unify, parse_name_cfg, arg_parser
from postprocessing.tokenizer import Tokenizer
from utils.evaluation import run_probe

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

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
    probe_trn_loader, probe_val_loader, _, _ = get_data(args, is_probe = True)   

    global_epoch = 1
    for _run in range(args.num_runs):
        logging.info(f"Run {_run}")
        args.tokenizer = Tokenizer(args.tokenizer_path) if hasattr(args, "tokenizer_path") else "tokenizer.json"
        best_mlp_metric = np.inf

        pbar = tqdm(range(1, args.max_epoch + 1))
        for _ in pbar:
            log_dict, mlp_metric = run_probe(global_epoch, None, probe_trn_loader, probe_val_loader, descriptors[args.mlpprob.dataset], device, args)
            if best_mlp_metric > mlp_metric:
                best_mlp_metric = mlp_metric
                
            pbar.set_postfix(log_dict)
            wandb.log(log_dict, step=global_epoch)
            global_epoch += 1

    wandb.finish()

def run_dataset_main(args, dataset = None, seed = None):
    args.dataset = dataset if dataset is not None else args.dataset
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

def run(args):
    run_dataset_main(args)  

if __name__ == "__main__":
    _, args = arg_parser(default_file="cfgs/experiments/baseline.yaml")
    args = args_unify(args_canonize(args))
    run()

