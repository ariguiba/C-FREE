import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils.misc import args_canonize, args_unify, arg_parser
import finetune, pretrain

# Define the datasets to run on. You can choose from the following datasets:
DRUGS_DATASETS = ['drugs_ip', 'drugs_ea', 'drugs_chi']
KRAKEN_DATASETS = ['kraken_l', 'kraken_b5', 'kraken_burb5', 'kraken_burl'] 
MOLECULENET_DATASETS = ['SIDER', 'BACE', 'BBBP', 'ClinTox', 'ToxCast', 'Tox21', 'HIV', 'MUV']
OTHER_DATASETS = ['exp'], ['zinc'], ['geom'], ['chiro'] 

# Setup the mode to "finetune" or "pretrain" and choose the config file to use. 
MODE = "pretrain"
CONFIG_FILE = "cfgs/pretrain-default.yaml"

# Setup for multiple run, use the same config file for different datasets or run the same dataset with multiple seeds. Set MANY_DATASETS to True to run multiple datasets, and set the list of datasets and seeds to use. 
# If MANY_DATASETS is False, it will just run the config file as is (which can be set to any dataset and seed). 
# If MANY_DATASETS is True, it will ignore the dataset and seed in the config file and use the ones specified above.
MANY_DATASETS = False
DATASETS = KRAKEN_DATASETS
SEED = [1]

if __name__ == "__main__":
    _, args = arg_parser(default_file=CONFIG_FILE)
    args = args_unify(args_canonize(args))

    if MANY_DATASETS:
        for dataset in DATASETS:
            for seed in SEED:
                print(f"Running evaluation on {dataset}")
                if MODE == "pretrain":
                    pretrain.run_dataset_main(args, dataset, seed)
                else:
                    finetune.run_dataset_main(args, dataset, seed)
    else:
        pretrain.run_dataset_main(args, None, None) if MODE == "pretrain" else finetune.run_dataset_main(args, None, None)

