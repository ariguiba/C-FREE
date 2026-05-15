#!/bin/bash
#SBATCH --job-name=3d-abl-molmix
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00 
#SBATCH --qos=student_high_res
#SBATCH --partition=slowlane
#SBATCH --gres=gpu:1     
#SBATCH --mem=64G        
#SBATCH --cpus-per-task=4


log_prefix="${1}"

if [ -z "$log_prefix" ]; then
  echo "Experiment: $0 <log_prefix>"
  exit 1
fi

module load CUDA
module load Miniconda3
module list

# conda init
#source ~/anaconda3/etc/profile.d/conda.sh
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate pyg

which conda
conda info --envs
echo "Conda environment: $(conda info --envs | grep '^*')"

export PATH=/mnt/beegfs/home/st162699/.conda/envs/pyg/bin:$PATH
echo "PYTHON: $(which python)"
python --version

# echo "installing environment"
# pip install torchlars


echo "Running experiment now"
# python3 prepare_data.py > "${log_prefix}-preprocess.log" 2>&1 &
python main-fft-mod-ablation-molmix.py > logs/"${log_prefix}.log" 2>&1 
