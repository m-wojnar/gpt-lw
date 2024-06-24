#!/bin/bash

BASE_DIR="/workspace/gpt-lw"
VENV_DIR="${BASE_DIR}/venv"

cd ${BASE_DIR}
source ${VENV_DIR}/bin/activate

python train.py "$@" --loss_weighting negexp_relpos
python train.py "$@" --loss_weighting unweighted
