#!/bin/bash
#SBATCH --job-name=drugs_kraken_experiment_prep
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=4:00:00  # Adjust as needed
#SBATCH --partition=gpuB01 # Change to your cluster's partition
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --mem=32G        # Adjust memory as needed
#SBATCH --cpus-per-task=4 # Adjust CPUs as needed


log_prefix="${1}"

if [ -z "$log_prefix" ]; then
  echo "Experiment: $0 <log_prefix>"
  exit 1
fi

# conda init
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pyg

python3 prepare_data.py --dataset drugs --policies one-ego-nets-and-comp > "${log_prefix}-preprocess.log" 2>&1 &
# Optional delay
# sleep 10
# python3 main.py --cfg cfgs/experiments/drugs-sampled-0.1.yaml > "${log_prefix}-experiment-1.log" 2>&1 
