#!/bin/bash

BASE_DIR="${BASE_DIR:=/workspace/gpt-lw}"
VENV_DIR="${VENV_DIR:=$BASE_DIR/venv}"

cd ${BASE_DIR}
source ${VENV_DIR}/bin/activate

python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting unweighted --run_name llama_wiki_mini_short
# python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/llama_wiki_mini/analysis/abs_random_weights.npy --run_name llama_wiki_mini_short_abs_random
# python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/llama_wiki_mini/analysis/abs_weights.npy --run_name llama_wiki_mini_short_abs
# python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/llama_wiki_mini/analysis/abs_weights_4.npy --run_name llama_wiki_mini_short_abs_4
# python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/llama_wiki_mini/analysis/abs_weights_8.npy --run_name llama_wiki_mini_short_abs_8
# python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/llama_wiki_mini/analysis/rel_weights.npy --run_name llama_wiki_mini_short_rel
# python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/llama_wiki_mini/analysis/rel_weights_4.npy --run_name llama_wiki_mini_short_rel_4
# python train.py --gpt_config configs/gpt/short_ctx.yaml --optimizer_config configs/optimizer/small_lr.yaml --train_config configs/train/base_txt.yaml --loss_weighting runs/llama_wiki_mini/analysis/rel_weights_8.npy --run_name llama_wiki_mini_short_rel_8

# stop pod
runpodctl stop pod $RUNPOD_POD_ID