#!/bin/bash
#SBATCH --job-name=extract_geom
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=2:00:00  # Adjust as needed
#SBATCH --partition=slowlane # Change to your cluster's partition
#SBATCH --mem=64G        # Adjust memory as needed
#SBATCH --cpus-per-task=4 # Adjust CPUs as needed

log_prefix="extract_geom"

echo "Extracting data now"
tar -xvf data/raw_data/geom/raw/rdkit_folder.tar.gz > logs/"${log_prefix}.log" 2>&1 
