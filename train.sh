#!/bin/bash

BASE_DIR="/workspace/gpt-lw"
VENV_DIR="${BASE_DIR}/venv"

cd ${BASE_DIR}
source ${VENV_DIR}/bin/activate

# python train.py "$@" --gpt_config configs/gpt/long_ctx.yaml --optimizer_config configs/optimizer/base.yaml --train_config configs/train/cfg3b.yaml --loss_weighting negexp_relpos --run_name cf3g3b_negexp_relpos
python train.py "$@" --gpt_config configs/gpt/long_ctx.yaml --optimizer_config configs/optimizer/base.yaml --train_config configs/train/cfg3b.yaml --loss_weighting unweighted --run_name cf3g3b_half_unweighted