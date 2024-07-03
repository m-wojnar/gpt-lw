#!/bin/bash

# Create directory if it doesn't exist
mkdir -p wikipedia

N=1

# Generate file list with zero-padded numbers
FILE_LIST=()
for i in $(seq 0 $N); do
    FILE_LIST+=("train-$(printf "%05d" $i)-of-00041.parquet")
done

# Base URL for downloading files
BASE_URL="https://huggingface.co/datasets/legacy-datasets/wikipedia/resolve/main/data/20220301.en"

# Download files
for FILE in "${FILE_LIST[@]}"; do
    wget "${BASE_URL}/${FILE}" -P wikipedia/
done