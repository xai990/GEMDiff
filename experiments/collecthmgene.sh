#!/bin/bash
GENE_DIR="coregene"
output_file="${GENE_DIR}/allhallmark"
temp="tempgene"
> "$output_csv"

for file in "$GENE_DIR"/*.gmt; do
    
    awk -F, '
        NR==1 {print $0}  # Print the header (first line)
        NR>1 {for (i=1; i<=NF; i++) printf ","$i; printf "\n"}  # Print the remaining rows as is
    ' "$file" >> "$temp"
    echo "Processed: $file"
done

awk '{
    for (i=3; i<=NF; i++) {
        printf "%s\t", $i  # Start from the 3rd column and concatenate columns with tabs
    }
} END { print "" }' "$temp" > "$output_file"

rm $temp