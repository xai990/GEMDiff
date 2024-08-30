#!/bin/bash 


directory="log/hallmark"
search_word=" experiemnt of silhouette score:"
output_csv="silhouette_score_hallmark.csv"


file_name="log.*"

> "$output_csv"

for FILE in $directory; do 
    find "$FILE" -type f -name "$file_name" | while read -r file_path; do
        echo "$file_path"
        grep "$search_word" "$file_path"| while read -r line; do
            csv_line=$(echo "$line" | sed 's/ /,/g')
            echo "$csv_line" >> "$output_csv"
        done
    done
done

