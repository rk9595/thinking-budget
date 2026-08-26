#!/usr/bin/env bash
# Runs unattended after training finishes: eval -> inference profile -> plots -> HF upload.
# Waits for an already-running train_grpo.py rather than starting one.
# Launch under tmux so it survives disconnects:
#   tmux new -d -s pipeline 'bash /workspace/pipeline.sh 2>&1 | tee /workspace/pipeline.log'
set -uo pipefail
cd /workspace/thinking-budget

export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${HF_TOKEN:?HF_TOKEN required}"
REPO="${REPO:-rk9595/thinking-budget-qwen1.5b-lcpo}"
DPH="${DPH:-1.086}"

echo "[pipeline] waiting for training to finish..."
while pgrep -f "train_grpo.py" >/dev/null; do sleep 60; done
echo "[pipeline] training process gone at $(date -u)"

# sort on the trailing step number only - the path itself contains a dash
CKPT=$(ls -d checkpoints/lcpo-max/checkpoint-* 2>/dev/null \
       | awk -F'checkpoint-' '{print $2, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -z "$CKPT" ]; then
  echo "[pipeline] FATAL: no checkpoint found"; exit 1
fi
echo "[pipeline] using $CKPT"

# Credit is tight on this run. Publish the adapter before the evals so a halt
# partway through the pipeline still leaves the artifact on HF; the final upload
# at the bottom adds the results on top.
echo "[pipeline] === early adapter upload ==="
python upload_hf.py --adapter "$CKPT" --repo "$REPO" --results /nonexistent \
  || echo "[pipeline] WARN: early upload failed, continuing"

# The 2026-08-22 base.json predates the overshoot metrics, so it cannot be
# compared against a max-variant run; recapture it (~10 min) rather than plot
# a half-empty chart.
if ! python -c "import json,sys; sys.exit(0 if 'mean_overshoot' in json.load(open('results/base.json'))['results'][0] else 1)" 2>/dev/null; then
  echo "[pipeline] === base eval (recapture with overshoot metrics) ==="
  python eval_budget.py --datasets math500 gsm8k \
    --budgets 256 512 1024 2048 --limit 100 --max-tokens 4096 \
    --out results/base.json
  echo "[pipeline] base eval exit=$?"
fi

echo "[pipeline] === trained eval ==="
python gpu_monitor.py --out results/gpu_eval.csv --interval 2 & MON=$!
python eval_budget.py --lora "$CKPT" --datasets math500 gsm8k \
  --budgets 256 512 1024 2048 --limit 100 --max-tokens 4096 \
  --out results/trained.json
echo "[pipeline] eval exit=$?"
kill $MON 2>/dev/null

echo "[pipeline] === checkpoint sweep (adherence vs training step) ==="
# Cheap reduced eval on intermediate checkpoints so the write-up shows a trend,
# not just a single endpoint. ~1 min each.
FINAL_STEP=$(echo "$CKPT" | awk -F'checkpoint-' '{print $2}')
# every point uses identical settings so the trend is comparable end to end
for STEP in 100 250 500 750 "$FINAL_STEP"; do
  C="checkpoints/lcpo-max/checkpoint-$STEP"
  if [ -d "$C" ] && [ ! -f "results/ckpt_$STEP.json" ]; then
    echo "[pipeline] sweep $C"
    python eval_budget.py --lora "$C" --datasets math500 \
      --budgets 256 512 1024 2048 --limit 50 --max-tokens 4096 \
      --out "results/ckpt_$STEP.json" 2>&1 | tail -3
  fi
done

if [ "${SKIP_PROFILE:-0}" = "1" ]; then
  echo "[pipeline] === inference profile SKIPPED (SKIP_PROFILE=1) ==="
else
  echo "[pipeline] === inference profile ==="
  python gpu_monitor.py --out results/gpu_profile.csv --interval 2 & MON=$!
  # run 1's profile hung silently and cost the plots + final upload; cap it
  timeout 2700 python profile_inference.py --lora "$CKPT" --dph "$DPH" \
    --budgets 256 512 1024 2048 --concurrency 1 8 32 128 \
    --compare-hf --out results/inference_profile.json
  RC=$?
  echo "[pipeline] profile exit=$RC"
  [ "$RC" = "124" ] && echo "[pipeline] WARN: profile hit the 45min timeout, continuing"
  kill $MON 2>/dev/null
fi

echo "[pipeline] === plots ==="
python plot_results.py --base results/base.json --trained results/trained.json --out-dir results
# profile_inference.py died silently on the 2026-08-22 run and took its plots
# with it; don't let a missing profile block the rest
if [ -f results/inference_profile.json ]; then
  python plot_inference.py --profile results/inference_profile.json \
    --gpu results/gpu_train.csv results/gpu_eval.csv results/gpu_profile.csv --out-dir results
else
  echo "[pipeline] WARN: no inference_profile.json, skipping inference plots"
fi
python plot_progress.py --results results --out-dir results

echo "[pipeline] === gpu summaries ==="
for f in results/gpu_*.csv; do echo "--- $f"; python gpu_monitor.py --summarize "$f"; done \
  | tee results/gpu_summary.txt

echo "[pipeline] === upload ==="
python upload_hf.py --adapter "$CKPT" --repo "$REPO" --results results

echo "PIPELINE_DONE $(date -u)"
