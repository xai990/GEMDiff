#!/bin/bash 

LOG_DIR="log/enrichment/hallmark"
GENE_DIR="coregene/corelist"
FILE_PATTERN="model40000.pt"


for GENE_SET in "$GENE_DIR"/*; do
    JOB_NAME="$(basename "$GENE_SET")"

    model=$(find "$LOG_DIR/$JOB_NAME" -type f -name "$FILE_PATTERN" -print -quit)
    if [ -n "$model" ]; then
        echo "The variable is not null."
    else
        echo "The variable is null."
    fi
done