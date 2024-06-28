#!/bin/bash
#SBATCH --job-name gpt-lw
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 40GB
#SBATCH --time 4:00:00
#SBATCH --account plgfastdnns-gpu-a100
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gres gpu
#SBATCH --output logs/gpt-lw.out
#SBATCH --error logs/gpt-lw.err

BASE_DIR="${PLG_GROUPS_STORAGE}/plggdaisnet/mwojnar/gpt-lw"
VENV_DIR="${BASE_DIR}/venv"
cd ${BASE_DIR}

module load Python/3.10.4
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT
source "${VENV_DIR}/bin/activate"

python train.py --gpt_config configs/gpt/long_ctx.yaml --optimizer_config configs/optimizer/base.yaml --train_config configs/train/base.yaml --loss_weighting ngram_1 --run_name cf3g3b_half_ngram_1
