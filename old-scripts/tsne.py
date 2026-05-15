import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

import numpy
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
torch.serialization.add_safe_globals([numpy.dtype])

import wandb as wandb
from preprocessing.dataloaders import get_data, get_dataset
from layers.ssl_models import get_sl_model
from preprocessing.utils.target_metric import get_target_metric
from utils.metrics import Evaluator, IsBetter
from utils.misc import args_canonize, args_unify, save_state, parse_name_cfg, log_epoch, arg_parser
from postprocessing.tokenizer import Tokenizer
from utils.evaluation import get_features_and_target
import postprocessing.labelextractor as labelextractor

# torch.autograd.set_detect_anomaly(True)

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)



def main(args, wandb):
    wandb.config.update(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # modes: create data, baseline (prop - raw features), ssl train, sl train
    ssl = args.ssl if hasattr(args, "ssl") else False    
    save_ckpt = hasattr(args, "save_ckpt") and args.save_ckpt
    load_ckpt = args.load_ckpt if hasattr(args, "load_ckpt")  else False
    pool_subgraphs = not (args.data.policy == 'original')

    trn_loader, val_loader, tst_loader, task = get_data(args, new_label=True) 
    model = get_sl_model(args, device) # we use target encoder for fine-tuning
    print(model)


    if load_ckpt:
        checkpoint = torch.load(f"checkpoints/{args.backbone_file}.pth", weights_only=False)
        state_dict = checkpoint["state_dict"]
        # new_state_dict = rename_for_2d(state_dict)
        new_state_dict = state_dict
        load_result = model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)
    
    myloader = trn_loader.loader

    # X_train, y_train = get_features_and_target(model, trn_loader.loader, device, with_subgraphs = pool_subgraphs, args = args)
    # X_val, y_val = get_features_and_target(model, val_loader.loader, device, with_subgraphs = pool_subgraphs, args = args)
    tokens_tf, tokens_2d, tokens_3d, y = get_features_and_target(model, myloader, device, with_subgraphs = pool_subgraphs, args = args, with_tokens=True)

    smiles = []
    for data in myloader:
        for smi in data.smiles:
            smiles.append(smi)

    extractor = labelextractor.MolecularLabelExtractor()
    new_labels = extractor.extract_all_labels(smiles)
    primary_functional_groups = new_labels['primary_functional_group']
    chirality = new_labels['has_chirality']
    aromatic_rings = new_labels['n_aromatic_rings']

    analyzer = labelextractor.EmbeddingAnalyzer(random_state=2)
    fig, metrics = analyzer.compare_embeddings(
        tokens_3d, 
        tokens_tf,
        chirality, #chirality, #primary_functional_groups
        label_type="Chirality", #"Functional Groups",
        method='umap'  # or 'umap'
    )   

    # fig, metrics = analyzer.compare_embeddings(
    #     tokens_2d, 
    #     tokens_tf,
    #     primary_functional_groups,
    #     label_type="Functional Groups",
    #     method='umap'  # 'umap' or 'tsne'
    # )   

    plt.savefig('chirality.png', dpi=300, bbox_inches='tight')
    print(f"Encoder NMI: {metrics['encoder']['nmi']:.3f}")
    print(f"Transformer NMI: {metrics['transformer']['nmi']:.3f}")
    print()


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

def run(args):
    run_dataset_main(args)  

if __name__ == "__main__":
    _, args = arg_parser(default_file="cfgs/tsne.yaml")
    args = args_unify(args_canonize(args))
    run(args)

