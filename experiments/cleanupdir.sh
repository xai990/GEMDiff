#!/bin/bash 

base_directory="log/enrichment/hallmark"

# Array of required filenames or patterns (you can use wildcards)
required_files=("*.pt" "*.png")

# Loop through all directories under the base directory
find "$base_directory" -mindepth 2 -type d | while read -r dir; do
    # Flag to indicate if required files are found
    found=false
    
    # Check for each required file pattern
    for pattern in "${required_files[@]}"; do
        # If at least one file matching the pattern is found, set the flag to true
        if find "$dir" -maxdepth 1 -type f -name "$pattern" | grep -q .; then
            found=true
            break
        fi
    done
    
    # If no required files are found, remove the directory
    if [ "$found" = false ]; then
        echo "Removing directory: $dir"
        rm -rf "$dir"
    fi
done