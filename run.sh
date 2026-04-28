#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export HF_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
NPROC=${NPROC:-2}
accelerate launch --config_file fsdp.yaml --num_processes $NPROC train.py
