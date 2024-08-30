#!/bin/bash 

CONFIG_FILE="configs/random/mrna_16.yaml"
SLURM_TEMPLATE="experiments/ddpm_perb.sh"
GENE_DIR="coregene/corelist"
JOB_DIR="jobs/"
LOG_DIR="/scratch/xai/DDPM-mRNA-augmentation-light/log/enrichment/hallmark"
FILE_PATTERN="model40000.pt"


for GENE_SET in "$GENE_DIR"/*; do
    # extrac the config parameters 
    CONFIG_PATH=$(realpath "$CONFIG_FILE")
    GENE_PATH=$(realpath "$GENE_SET")
    JOB_NAME="$(basename "$GENE_SET")"
    # create a new PBS script for this configration 
    echo $JOB_NAME
    SLURM_SCRIPT="ddpm_perb_${JOB_NAME}.sh"
    LOG_PATH="${LOG_DIR}/${JOB_NAME}"
    # replace the key for each sbatch job 
    sed -e "s|{{CONFIG_PATH}}|\"$CONFIG_PATH\"|" -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" "$SLURM_TEMPLATE" > "$SLURM_SCRIPT"
    sed -i "s|{{GENE_PATH}}|\"$GENE_PATH\"|" "$SLURM_SCRIPT"
    model=$(find "$LOG_DIR/$JOB_NAME" -type f -name "$FILE_PATTERN" -print -quit)
    MODEL_DIR=$(dirname "$model")
    sed -i "s|{{MODEL_DIR}}|\"$MODEL_DIR\"|" "$SLURM_SCRIPT"
    # submit the qsub job 
    sbatch "$SLURM_SCRIPT"
    mv "$SLURM_SCRIPT" "$JOB_DIR"
done

















# for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do 
#     # extrac the config parameters 
#     CONFIG_PATH=$(realpath "$CONFIG_FILE")
#     JOB_NAME=$(basename "$CONFIG_FILE" .yaml)
#     # create a new sh script for this configration 
#     SLURM_SCRIPT="ddpm_sample_${JOB_NAME}.sh"
#     LOG_PATH="log/${JOB_NAME}"
#     RESULT_PATH="results/${JOB_NAME}"
#     # replace the key for each SLURM job 
#     sed -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" "$SLURM_TEMPLATE" > "$SLURM_SCRIPT"   
#     # find the save model 
#     model=$(find "$LOG_DIR/$JOB_NAME" -type f -name "$FILE_PATTERN" -print -quit)
#     echo "Processing .pt file: $model"
#     sed -i "s|{{MODEL_PATH}}|\"$model\"|" "$SLURM_SCRIPT"
#     sed -i "s|{{RESULT_PATH}}|\"$RESULT_PATH\"|" "$SLURM_SCRIPT"
#     # submit the qsub job 
#     sbatch "$SLURM_SCRIPT"
#     mv "$SLURM_SCRIPT" "$JOB_DIR"
# done



# for GENE_SET in "$GENE_DIR"/*.gmt; do
#     for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do 
#         # extrac the config parameters 
#         CONFIG_PATH=$(realpath "$CONFIG_FILE")
#         GENE_PATH=$(realpath "$GENE_SET")
#         JOB_GENE=$(basename "$GENE_SET" .gmt)
#         JOB_CONF=$(basename "$CONFIG_FILE" .yaml)
#         JOB_NAME="${JOB_GENE}_${JOB_CONF}"
#         echo $JOB_NAME
#         # create a new PBS script for this configration 
#         PBS_SCRIPT="ddpm_sample_${JOB_NAME}.pbs"
#         LOG_PATH="log/${JOB_NAME}"
#         RESULT_PATH="results/${JOB_NAME}"
#         # replace the key for each pbs job 
#         sed -e "s|{{RESULT_PATH}}|$RESULT_PATH|" -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" "$PBS_TEMPLATE" > "$PBS_SCRIPT"   
#         # find the save model 
#         model=$(find "$LOG_DIR/$JOB_NAME" -type f -name "$FILE_PATTERN" -print -quit)
#         echo "Processing .pt file: $model"
#         sed -i "s|{{MODEL_PATH}}|\"$model\"|" "$PBS_SCRIPT"
#         # submit the qsub job 
#         qsub "$PBS_SCRIPT"

#         mv "$PBS_SCRIPT" "$JOB_DIR"
#     done
# done

