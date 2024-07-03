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

export BASE_DIR="${PLG_GROUPS_STORAGE}/plggdaisnet/mwojnar/gpt-lw"
export VENV_DIR="${BASE_DIR}/venv"

module load Python/3.10.4
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT

./train.sh
