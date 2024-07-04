#!/bin/bash

BASE_DIR="${BASE_DIR:=/workspace/gpt-lw}"
VENV_DIR="${VENV_DIR:=$BASE_DIR/venv}"

cd ${BASE_DIR}
source ${VENV_DIR}/bin/activate

python train.py --gpt_config configs/gpt/long_ctx.yaml --optimizer_config configs/optimizer/base.yaml --train_config configs/train/base_txt.yaml --loss_weighting unweighted --run_name llama_wiki_mini
# python train.py --gpt_config configs/gpt/long_ctx.yaml --optimizer_config configs/optimizer/base.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/wiki_mini/analysis/weights_pow_smoothed.npy --run_name wiki_mini_empirical_relpos_weighting

# stop pod
# runpodctl stop pod $RUNPOD_POD_ID