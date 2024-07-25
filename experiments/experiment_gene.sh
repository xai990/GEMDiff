#!/bin/bash 

LOG_DIR="logUAMP"
GENE_DIR="geneset_score"
BATCH_TEMPLATE="ddpm_gene.sh"
JOB_DIR="jobs"
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
            
            echo $JOB_NAME
            BATCH_JOB="geneselect_${JOB_NAME}.sh"
            LOG_PATH="${LOG_DIR}/${JOB_NAME}"
            # replace the key for each pbs job 
            sed -e "s|{{JOB_NAME}}|$JOB_NAME|" -e "s|{{LOG_PATH}}|\"$LOG_PATH\"|" -e "s|{{GENE_PATH}}|\"$GENE_PATH\"|" "$BATCH_TEMPLATE" > "$BATCH_JOB"
            sbatch "$BATCH_JOB"
            mv "$BATCH_JOB" "$JOB_DIR"
        done
    fi 
done