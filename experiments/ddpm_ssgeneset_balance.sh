#!/bin/bash

#SBATCH --job-name=silhouette       # Set the job name
#SBATCH --nodes 1
#SBATCH --tasks-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --mem 32gb
#SBATCH --time 24:00:00

# This should be the directory where you cloned the DDPM-mRNA-augmentation repository
DDPM_DIR="/scratch/xai/DDPM-mRNA-augmentation-light"

#Create conda environment from instructions in DDPM-mRNA-augmentation readme
module purge
module load anaconda3/2023.09-0
source activate DDIM 

# Move to the python package directory 
cd ${DDPM_DIR}
# config file path 
LOG_DIR="log/silhouette_geneset_balance"
GENE_DIR="geneset_score"
# GENE_PATH={{GENE_PATH}}
# Define the pattern to search for .egg-info directories
egg_info_pattern="*.egg-info"

if find . -maxdepth 1 -type d -name "$egg_info_pattern" | grep -q .; then
    echo "The .egg-info directory exists. Skipping 'pip install -e .'"
else
    # install and build the package environment 
    pip install -e .
fi

for sub in "$GENE_DIR"/*; do
    # Check if the item is a file and not a directory
    if [ -d "$sub" ]; then
        for GENE_SET in "$sub"/*; do
            GENE_PATH=$(realpath "$GENE_SET")
            if [[ "${filename,,}" =~ \.gmt$ ]]; then
                JOB_NAME="$(basename "$GENE_SET" .gmt)"
            else
                JOB_NAME="$(basename "$GENE_SET")"
            fi
            LOG_PATH="${LOG_DIR}/${JOB_NAME}"
            python scripts/gene.py --dir $LOG_PATH --gene_set $GENE_PATH --balance
        done
    fi
done