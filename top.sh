#!/bin/bash

# Function to extract the accuracy value from the file name
extract_accuracy() {
    echo "$1" | grep -oP '(?<=_acc_).*(?=\.h5)'
}

# Create an associative array to store the file names and their accuracies
declare -A files_with_accuracies

# Loop through all files in the current directory
for file in *.h5; do
    # Extract the accuracy from the file name
    accuracy=$(extract_accuracy "$file")

    # Store the file name and accuracy in the associative array
    files_with_accuracies["$file"]="$accuracy"
done

# Sort the file names based on their accuracies in descending order
sorted_files=($(
    for file in "${!files_with_accuracies[@]}"; do
        printf "%s\t%s\n" "${files_with_accuracies[$file]}" "$file"
    done | sort -rn
))

# Print the top 5 models
echo "Top 5 models:"
for ((i = 0; i < 6 && i < ${#sorted_files[@]}; i++)); do
    accuracy=$(echo "${sorted_files[$i]}" | awk '{print $1}')
    file=$(echo "${sorted_files[$i]}" | awk '{print $2}')
    echo "$file (Accuracy: $accuracy)"
done