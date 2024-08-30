#!/bin/bash 

CONFIG_FILE="configs/random/mrna_16.yaml"
SLURM_TEMPLATE="experiments/ddpm_train.sh"
GENE_DIR="coregene/corelist"
JOB_DIR="jobs/"
LOG_DIR="log/enrichment/hallmark"

for GENE_SET in "$GENE_DIR"/*; do
    # extrac the config parameters 
    CONFIG_PATH=$(realpath "$CONFIG_FILE")
    GENE_PATH=$(realpath "$GENE_SET")
    JOB_NAME="$(basename "$GENE_SET")"
    # create a new PBS script for this configration 
    echo $JOB_NAME
    SLURM_SCRIPT="ddpm_train_${JOB_NAME}.sh"
    LOG_PATH="${LOG_DIR}/${JOB_NAME}"
    # replace the key for each sbatch job 
    sed -e "s|{{CONFIG_PATH}}|\"$CONFIG_PATH\"|" -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" "$SLURM_TEMPLATE" > "$SLURM_SCRIPT"
    sed -i "s|{{GENE_PATH}}|\"$GENE_PATH\"|" "$SLURM_SCRIPT"
    # submit the qsub job 
    sbatch "$SLURM_SCRIPT"
    mv "$SLURM_SCRIPT" "$JOB_DIR"
done



# CONFIG_DIR="configs/random/"
# SLURM_TEMPLATE="experiments/ddpm_train.sh"
# GENE_PATH="Random"
# JOB_DIR="jobs/"

# for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do 
#     # extrac the config parameters 
#     CONFIG_PATH=$(realpath "$CONFIG_FILE")
#     JOB_NAME=$(basename "$CONFIG_FILE" .yaml)
#     # create a new PBS script for this configration 
#     echo $JOB_NAME
#     SLURM_SCRIPT="ddpm_${JOB_NAME}.sh"
#     LOG_PATH="log/${JOB_NAME}"
#     # replace the key for each sbatch job 
#     sed -e "s|{{CONFIG_PATH}}|\"$CONFIG_PATH\"|" -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" "$SLURM_TEMPLATE" > "$SLURM_SCRIPT"
#     sed -i "s|{{GENE_PATH}}|\"$GENE_PATH\"|" "$SLURM_SCRIPT"
#     # submit the qsub job 
#     sbatch "$SLURM_SCRIPT"
#     mv "$SLURM_SCRIPT" "$JOB_DIR"
# done

