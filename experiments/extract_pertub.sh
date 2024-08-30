# #!/bin/bash
# directory="log/enrichment"
# output_csv="core_genes.csv"

# file_name="log.*"
# > "$output_csv"

# echo "GeneSet,Mean,Standard Deviation,Identified Genes" > "$output_csv"
# for FILE in $directory; do 
#     find "$FILE" -type f -name "$file_name" | while read -r file_path; do
#         echo "$file_path"
#         geneset=$(grep -oP '(?<=log/enrichment/)[^/]+' "$file_path")
#         mean=$(grep -oP 'The mean between real and perturb data data \K[0-9]+\.[0-9]+' "$file_path")
#         std_dev=$(grep -oP 'The standard deviation between real and perturb data data \K[0-9]+\.[0-9]+' "$file_path")
#         identified_genes=$(grep -oP "(?<=The indentified genes are: Index\(\[)[^]]+" "$file_path" | sed "s/', '/,/g" | sed "s/'//g")
#         if [ -n "$identified_genes" ] && [ -n "$std_dev" ] && [ -n "$mean" ]; then
#             echo "$geneset,$mean,$std_dev,$identified_genes" >> "$output_csv"
#         fi
#     done
# done 

#!/bin/bash
directory="log/enrichment/hallmark"
output_csv="hallmark_gene.csv"

file_name="log.*"
> "$output_csv"

# Header for the CSV file
echo "GeneSet,Mean,Standard Deviation,Identified Genes" > "$output_csv"

# Loop through each file found in the directory
find "$directory" -type f -name "$file_name" | while read -r file_path; do
    
    # Extract the relevant information
    geneset=$(grep -oP '(?<=log/enrichment/)[^/]+' "$file_path")
    mean=$(grep -oP 'The mean between real and perturb data data \K[0-9]+\.[0-9]+' "$file_path")
    std_dev=$(grep -oP 'The standard deviation between real and perturb data data \K[0-9]+\.[0-9]+' "$file_path")
    
    identified_genes=$(grep -oP "(?<=The indentified genes are: Index\(\[)[^]]+" "$file_path" | sed "s/', '/,/g" | sed "s/'//g")
    
    # Check if all required values are present
    if [ -n "$geneset" ] && [ -n "$identified_genes" ] && [ -n "$std_dev" ] && [ -n "$mean" ]; then
        # Combine the values into a single line and append to the CSV file
        echo "$geneset,$mean,$std_dev,$identified_genes" >> "$output_csv"
    fi
done
