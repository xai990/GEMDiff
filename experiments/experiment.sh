#!/bin/bash 

CONFIG_FILE="configs/mrna_16.yaml"
SLURM_TEMPLATE="ddpm_train.sh"
GENE_DIR="geneset/perturb"
JOB_DIR="jobs"

for GENE_SET in "$GENE_DIR"/*; do
    # extrac the config parameters 
    GENE_PATH=$(realpath "$GENE_SET")
    CONFIG_PATH=$(realpath "$CONFIG_FILE")
    JOB_CONF=$(basename "$CONFIG_FILE" .yaml)
    JOB_GENE=$(basename "$GENE_SET")
    JOB_NAME="${JOB_GENE}_${JOB_CONF}"
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
