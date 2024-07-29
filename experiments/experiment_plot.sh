#!/bin/bash 

LOG_DIR="log"
FILE_PATTERN="*.npz"
SLURM_TEMPLATE="experiments/ddpm_plot.sh"
JOB_DIR="jobs"
CONFIG_DIR="configs/random/"
RESULTS_DIR="results/"

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do 
    # extrac the config parameters 
    CONFIG_PATH=$(realpath "$CONFIG_FILE")
    JOB_NAME=$(basename "$CONFIG_FILE" .yaml)
    # create a new sh script for this configration 
    SLURM_SCRIPT="ddpm_sample_${JOB_NAME}.sh"
    LOG_PATH="log/${JOB_NAME}"
    RESULT_PATH="results/${JOB_NAME}"
    # replace the key for each SLURM job 
    sed -e "s|{{CONFIG_PATH}}|\"$CONFIG_PATH\"|" -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" "$SLURM_TEMPLATE" > "$SLURM_SCRIPT"   
    # find the save sample 
    sample=$(find "$RESULTS_DIR/$JOB_NAME" -type f -name "$FILE_PATTERN" -print -quit)
    echo "Processing .npz file: $sample"
    sed -i "s|{{SAMPLE_PATH}}|\"$sample\"|" "$SLURM_SCRIPT"
    sed -i "s|{{RESULT_PATH}}|\"$RESULT_PATH\"|" "$SLURM_SCRIPT"
    # submit the qsub job 
    sbatch "$SLURM_SCRIPT"
    mv "$SLURM_SCRIPT" "$JOB_DIR"
done


