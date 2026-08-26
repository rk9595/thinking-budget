#!/usr/bin/env bash
# Provision a vast.ai GPU, run the full training pipeline on it, pull results back, destroy it.
#
#   export VAST_API_KEY=...      # from https://cloud.vast.ai/account/  (account must have credit)
#   export HF_TOKEN=... WANDB_API_KEY=...
#   bash rent_and_run.sh                 # auto-destroys after fetching results
#   bash rent_and_run.sh --keep          # leaves the box running
#
# Cost guard: refuses any offer above MAX_DPH.
set -euo pipefail

MAX_DPH="${MAX_DPH:-1.30}"           # $/hr ceiling
DISK_GB="${DISK_GB:-80}"
IMAGE="${IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel}"
MAX_STEPS="${MAX_STEPS:-1000}"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

VAST=".venv/bin/vastai"
: "${VAST_API_KEY:?set VAST_API_KEY}"
: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
$VAST set api-key "$VAST_API_KEY" >/dev/null

echo "==> Balance"
$VAST show user --raw | python3 -c 'import json,sys; print("  credit: $%.2f" % json.load(sys.stdin).get("credit",0))'

echo "==> Cheapest A100 80GB under \$$MAX_DPH/hr"
OFFER=$($VAST search offers "num_gpus=1 rentable=true gpu_ram>=79 disk_space>=$DISK_GB" \
        -o 'dph+' --raw | python3 -c "
import json,sys
for o in json.load(sys.stdin):
    if o['dph_total'] <= $MAX_DPH and ('A100' in o['gpu_name'] or 'H100' in o['gpu_name']):
        print(o['id'], round(o['dph_total'],3), o['gpu_name'].replace(' ','_')); break
")
[ -z "$OFFER" ] && { echo "No offer under \$$MAX_DPH/hr. Raise MAX_DPH."; exit 1; }
read -r OFFER_ID DPH GPU <<<"$OFFER"
echo "  offer $OFFER_ID  \$$DPH/hr  $GPU"
echo "  est. cost for ~30h: \$$(python3 -c "print(round($DPH*30,2))")"

echo "==> Creating instance"
INST=$($VAST create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK_GB" \
        --ssh --direct --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')
echo "  instance $INST"

cleanup() {
  if [ "$KEEP" = "0" ]; then
    echo "==> Destroying instance $INST"
    $VAST destroy instance "$INST" || echo "  DESTROY FAILED - kill it manually at https://cloud.vast.ai/instances/"
  else
    echo "==> Keeping instance $INST (billing continues; destroy with: $VAST destroy instance $INST)"
  fi
}
trap cleanup EXIT

echo "==> Waiting for instance to come up"
for i in $(seq 1 60); do
  STATUS=$($VAST show instance "$INST" --raw | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("actual_status",""), d.get("ssh_host",""), d.get("ssh_port",""))')
  read -r ST HOST PORT <<<"$STATUS"
  [ "$ST" = "running" ] && [ -n "$HOST" ] && break
  echo "  [$i/60] status=$ST"; sleep 15
done
[ "$ST" != "running" ] && { echo "Instance never started"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$HOST"
echo "  ssh: $SSH"

echo "==> Syncing repo"
until $SSH true 2>/dev/null; do echo "  waiting for sshd"; sleep 10; done
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT" \
  --exclude .venv --exclude data --exclude checkpoints --exclude results --exclude __pycache__ \
  ./ "root@$HOST:/workspace/thinking-budget/"

echo "==> Running pipeline (streaming; this is the long part)"
$SSH "export HF_TOKEN=$HF_TOKEN WANDB_API_KEY=$WANDB_API_KEY; \
      cd /workspace/thinking-budget && \
      pip install -q -r requirements.txt vllm && \
      python -m pytest tests/ -q && \
      python prepare_data.py --num-samples 8000 --out data/train && \
      python eval_budget.py --out results/base.json && \
      python train_grpo.py --output-dir checkpoints/lcpo-max --max-steps $MAX_STEPS && \
      python eval_budget.py --lora checkpoints/lcpo-max/checkpoint-$MAX_STEPS --out results/trained.json"

echo "==> Fetching results + adapter"
mkdir -p results checkpoints
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT" \
  "root@$HOST:/workspace/thinking-budget/results/" results/
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT" \
  "root@$HOST:/workspace/thinking-budget/checkpoints/lcpo-max/checkpoint-$MAX_STEPS/" \
  "checkpoints/lcpo-max/checkpoint-$MAX_STEPS/"

echo "==> Done. Plotting."
.venv/bin/python plot_results.py
