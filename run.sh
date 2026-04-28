#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export WANDB_PROJECT=${WANDB_PROJECT:-speaker-separator-pretrain}
NPROC=${NPROC:-2}
PORT=${PORT:-$((29500 + RANDOM % 1000))}
accelerate launch --num_processes $NPROC --main_process_port $PORT train.py
