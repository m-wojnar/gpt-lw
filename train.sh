#!/bin/bash

BASE_DIR="${BASE_DIR:=/workspace/gpt-lw}"
VENV_DIR="${VENV_DIR:=$BASE_DIR/venv}"

cd "${BASE_DIR}"
source "${VENV_DIR}/bin/activate"

# python train.py --gpt_config configs/gpt/short_ctx_10m.yaml --optimizer_config configs/optimizer/adam_cosine.yaml --train_config configs/train/train_txt.yaml --loss_weighting unweighted --run_name llama_wiki_short_10m
python train.py --gpt_config configs/gpt/short_ctx_10m.yaml --optimizer_config configs/optimizer/adam_cosine.yaml --train_config configs/train/train_txt.yaml --loss_weighting reciprocal --loss_weighting_a="$1" --loss_weighting_b="$2" --run_name "llama_wiki_short_10m_abs_rec_$1_$2"

# stop pod
#runpodctl stop pod $RUNPOD_POD_ID