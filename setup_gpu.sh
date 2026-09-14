#!/usr/bin/env bash
# One-shot setup on a rented GPU box (runpod/vast, CUDA image, A100 80GB).
# Usage: bash setup_gpu.sh
set -euo pipefail

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
pip check

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Fail fast on the cheap stuff before burning GPU hours.
python -m pytest tests/ -q

echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  python run_comparison.py --out-dir results/control-2026-09-08"
