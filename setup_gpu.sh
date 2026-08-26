#!/usr/bin/env bash
# One-shot setup on a rented GPU box (runpod/vast, CUDA image, A100 80GB).
# Usage: bash setup_gpu.sh
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install vllm
# flashinfer 0.6.x fails to import on Python 3.11 and vLLM imports it unguarded;
# removing the package is half the fix, VLLM_USE_FLASHINFER_SAMPLER=0 is the other.
pip uninstall -y flashinfer-python || true

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Fail fast on the cheap stuff before burning GPU hours.
python -m pytest tests/ -q

# HF_TOKEN avoids the unauthenticated download rate limit; WANDB_API_KEY for logging.
: "${HF_TOKEN:?set HF_TOKEN before running}"
: "${WANDB_API_KEY:?set WANDB_API_KEY before running}"

# must match --length-reward at train time; train_grpo.py verifies it
export TB_LENGTH_REWARD="${TB_LENGTH_REWARD:-exact}"
python prepare_data.py --num-samples 8000 --out data/train

echo "Setup done. Next:"
echo "  python eval_budget.py --out results/base.json          # baseline, ~20 min"
echo "  python train_grpo.py --output-dir checkpoints/lcpo-exact --max-steps 700"
