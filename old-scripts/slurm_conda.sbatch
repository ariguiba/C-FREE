#!/bin/bash
#SBATCH --job-name=conda-test
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=12:00:00  # Adjust as needed
#SBATCH --partition=slowlane # Change to your cluster's partition
#SBATCH --mem=64G        # Adjust memory as needed
#SBATCH --cpus-per-task=4 # Adjust CPUs as needed

export BASH_ENV=/mnt/nfs/lmod/lmod/init/bash

export MODULEPATH=/mnt/nfs/lmod/lmod/modulefiles/Core:/mnt/nfs/easybuild/modules/all

log_prefix="${1}"

if [ -z "$log_prefix" ]; then
  echo "Experiment: $0 <log_prefix>"
  exit 1
fi

# Unload all modules, load the module(s) you need and initialize Conda
module purge
module load Miniconda3
module list
# source ${EBROOTMINICONDA3}/bin/activate

# conda init
#source ~/anaconda3/etc/profile.d/conda.sh
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate pyg
conda list --export > packages.txt
conda create --name new-pyg --file packages.txt


which conda
conda info --envs

echo "PYTHON: $(which python)"
python --version
