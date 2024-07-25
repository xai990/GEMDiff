#!/bin/bash 

CONFIG_DIR="configs"
SLURM_TEMPLATE="experiments/ddpm_train.sh"
GENE_PATH="Random"
JOB_DIR="jobs/"

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do 
    # extrac the config parameters 
    CONFIG_PATH=$(realpath "$CONFIG_FILE")
    JOB_NAME=$(basename "$CONFIG_FILE" .yaml)
    # create a new PBS script for this configration 
    echo $JOB_NAME
    SLURM_SCRIPT="ddpm_${JOB_NAME}.sh"
    LOG_PATH="log/${JOB_NAME}"
    # replace the key for each sbatch job 
    sed -e "s|{{CONFIG_PATH}}|\"$CONFIG_PATH\"|" -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" "$SLURM_TEMPLATE" > "$SLURM_SCRIPT"
    sed -i "s|{{GENE_PATH}}|\"$GENE_PATH\"|" "$SLURM_SCRIPT"
    # submit the qsub job 
    sbatch "$SLURM_SCRIPT"
    mv "$SLURM_SCRIPT" "$JOB_DIR"
done
