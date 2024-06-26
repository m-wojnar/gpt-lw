#!/bin/bash

BASE_DIR="/workspace/gpt-lw"
VENV_DIR="${BASE_DIR}/venv"

cd ${BASE_DIR}

# Create a virtual environment
python -m venv venv
source ${VENV_DIR}/bin/activate
pip install -U pip setuptools wheel

# Install the requirements
pip install -r requirements.txt

# Generate the dataset
python generate_dataset.py

# Train the model
./train.sh "$@"
