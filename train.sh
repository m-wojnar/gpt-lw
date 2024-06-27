#!/bin/bash

BASE_DIR="/workspace/gpt-lw"
VENV_DIR="${BASE_DIR}/venv"

cd ${BASE_DIR}
source ${VENV_DIR}/bin/activate

python train.py "$@" --gpt_config configs/gpt/base.yaml --optimizer_config configs/optimizer/base.yaml --train_config configs/train/base.yaml --loss_weighting unweighted --run_name cf3g3b_half_clean