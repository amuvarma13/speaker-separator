#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export WANDB_PROJECT=${WANDB_PROJECT:-speaker-separator-pretrain}
export HF_HOME=${HF_HOME:-/workspace/.hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/workspace/.hf_cache/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/workspace/.hf_cache/transformers}
NPROC=${NPROC:-2}
PORT=${PORT:-$((29500 + RANDOM % 1000))}
accelerate launch --num_processes $NPROC --main_process_port $PORT train_separate.py
