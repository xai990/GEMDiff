#!/bin/bash 

directory="coregene/corelist"

output_csv="hallmark_set.csv"
> "$output_csv"


for FILE in "$directory"/*; do 
    echo "Processing file: $(basename "$FILE")"
    cat "$FILE" >> "$output_csv"
    # echo -e "\n" >> "$output_file"
done
echo "All files have been concatenated into $output_file."