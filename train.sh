#!/bin/bash

BASE_DIR="/workspace/gpt-lw"
VENV_DIR="${BASE_DIR}/venv"

cd ${BASE_DIR}
source ${VENV_DIR}/bin/activate

python train.py --gpt_config configs/gpt/long_ctx.yaml --optimizer_config configs/optimizer/base.yaml --train_config configs/train/base_txt.yaml --loss_weighting unweighted --run_name wiki_mini