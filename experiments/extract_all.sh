#!/bin/bash

directory="log/enrichment/hallmark"
output_csv="hallmark_allgene.csv"

file_name="log.*"
> "$output_csv"

echo "GeneSet,Gene,Real mean,perturbed mean,Difference" > "$output_csv"
counter=0
# Loop through each file found in the directory
find "$directory" -type f -name "$file_name" | while read -r file_path; do
    counter=$((counter + 1))
    if [ $((counter % 2)) -ne 0 ]; then
        continue
    fi
    echo "$file_path"
    # Extract the relevant information
    geneset=$(grep -oP '(?<=log/enrichment/hallmark/)[^/]+' "$file_path" | head -n 1)
    real_mean=$(grep -Pzo  "(?s)The real data.*?\-- script_util" "$file_path" | 
                head -z -n 1 |
                tr -d '\n' |
                sed "s/The real data //" |
                sed "s/-- script_util//g" |
                sed "s/'//g" | 
                sed "s/\[//g" | sed "s/\]//g" |
                tr -s ' ' '\n'
                )
    perturb_mean=$(grep -Pzo "(?s)The perturb data.*?\-- script_util" "$file_path" |
                head -z -n 1 |
                tr -d '\n' |
                sed "s/The perturb data //" |
                sed "s/-- script_util//g" |
                sed "s/'//g" | 
                sed "s/\[//g" | sed "s/\]//g" |
                tr -s ' ' '\n'
                )
    # difference=$(grep -A 2 "The differences between real and perturb data" "$file_path" | head -n 3 | sed "s/\] -- script_util//g" | sed "s/The differences between real and perturb data \[//g" | tr -d '\n' | tr -s ' ' '\n')
    difference=$(grep -Pzo "(?s)The differences between real and perturb data.*?\-- script_util" "$file_path" |
                head -z -n 1 |
                tr -d '\n' |
                sed "s/The differences between real and perturb data //" |
                sed "s/-- script_util//g" |
                sed "s/'//g" | 
                sed "s/\[//g" | sed "s/\]//g" |
                tr -s ' ' '\n'
                )
    all_genes=$(grep -Pzo "(?s)The selected genes are:.*?\) -- dataset" "$file_path" |
                head -z -n 1 |
                tr -d '\n' |
                sed "s/The selected genes are: Index(//" |
                sed "s/ dtype='object') -- dataset//g" | 
                sed "s/'//g" | 
                sed "s/\[//g" | sed "s/\]//g" |
                sed 's/, */,/g' |
                tr -s ',' '\n'
                )
    # echo "$geneset,$real_mean,$perturb_mean,$difference,$all_genes"
    echo "$real_mean" > real_mean.txt
    # echo "$real_mean"
    echo "$perturb_mean" > perturb_mean.txt
    # echo "$perturb_mean"
    echo "$difference" > difference.txt
    # echo "$difference"
    echo "$all_genes" > all_genes.txt
    # echo "$all_genes"
    yes "$geneset" | head -n $(wc -l < real_mean.txt) > geneset.txt
    sed -i '1{/^[[:space:]]*$/d}' real_mean.txt
    sed -i '1{/^[[:space:]]*$/d}' perturb_mean.txt
    sed -i '1{/^[[:space:]]*$/d}' difference.txt
    sed -i '1{/^[[:space:]]*$/d}' all_genes.txt

    paste -d ',' geneset.txt all_genes.txt real_mean.txt perturb_mean.txt difference.txt >> "$output_csv"
    rm real_mean.txt
    rm perturb_mean.txt
    rm difference.txt
    rm all_genes.txt
    rm geneset.txt
done
