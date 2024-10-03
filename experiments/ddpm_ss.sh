#!/bin/bash

#SBATCH --job-name=silhouette       # Set the job name
#SBATCH --nodes 1
#SBATCH --tasks-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --mem 32gb
#SBATCH --time 24:00:00

# This should be the directory where you cloned the DDPM-mRNA-augmentation repository
DDPM_DIR="/home/xai/Diffusion_Model/vscode/DDPM-mRNA-augmentation-light"

#Create conda environment from instructions in DDPM-mRNA-augmentation readme
module purge
module load anaconda3/2023.09-0
source activate DDIM 

# Move to the python package directory 
cd ${DDPM_DIR}
# config file path 
LOG_DIR={{LOG_PATH}}
CONFIG={{CONFIG}}
# GENE_PATH={{GENE_PATH}}
# Define the pattern to search for .egg-info directories
egg_info_pattern="*.egg-info"

if find . -maxdepth 1 -type d -name "$egg_info_pattern" | grep -q .; then
    echo "The .egg-info directory exists. Skipping 'pip install -e .'"
else
    # install and build the package environment 
    pip install -e .
fi
for ((i=0;i<1000;i++))
do 
    LOG_PATH="${LOG_DIR}/${i}"
    python scripts/gene.py --dir $LOG_PATH --random --config $CONFIG
done