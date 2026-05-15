import wandb
from utils.misc import Config, args_canonize, args_unify, create_nested_dict
from main import main

if __name__ == '__main__':
    # Load the baseline YAML config
    default_cfg_path = "cfgs/experiments/prop.yaml"
        
    wandb.init(mode="online",)
    wandb_config = create_nested_dict(wandb.config)
    args = args_canonize(wandb_config)
    config = Config()
    config.load(default_cfg_path, recursive=True)
    config.update(args)
    config = args_unify(config)
    main(config, wandb)
