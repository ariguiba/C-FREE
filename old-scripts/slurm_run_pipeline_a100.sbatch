#!/bin/bash
#SBATCH --job-name=molmix-1conf
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00 
#SBATCH --partition=slowlane
#SBATCH --nodelist=aisa-gpuA01
#SBATCH --gres=gpu:1     
#SBATCH --mem=64G        
#SBATCH --cpus-per-task=4

export BASH_ENV=/mnt/nfs/lmod/lmod/init/bash

export MODULEPATH=/mnt/nfs/lmod/lmod/modulefiles/Core:/mnt/nfs/easybuild/modules/all

log_prefix="${1}"

if [ -z "$log_prefix" ]; then
  echo "Experiment: $0 <log_prefix>"
  exit 1
fi

# Unload all modules, load the module(s) you need and initialize Conda
module purge
module load CUDA
module load Miniconda3
module list
# source ${EBROOTMINICONDA3}/bin/activate

source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate pyg

which conda
conda info --envs
echo "Conda environment: $(conda info --envs | grep '^*')"

# export PATH=/mnt/nfs/home/st162699/.conda/envs/pyg/bin:$PATH
# echo "PYTHON: $(which python)"
# python --version


echo "Running experiment now"
# python3 prepare_data.py > "${log_prefix}-preprocess.log" 2>&1 &
python main.py > logs/"${log_prefix}.log" 2>&1 
