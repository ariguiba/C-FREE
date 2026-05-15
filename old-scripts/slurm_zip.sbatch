#!/bin/bash
#SBATCH --job-name=zip_project
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=4:00:00  # Adjust as needed
#SBATCH --partition=slowlane # Change to your cluster's partition
#SBATCH --mem=64G        # Adjust memory as needed
#SBATCH --cpus-per-task=4 # Adjust CPUs as needed

log_prefix="extract_geom"

echo "Creating archive data now"
tar -czf data-zipped.tar.gz data/ > logs/"${log_prefix}.log" 2>&1 
