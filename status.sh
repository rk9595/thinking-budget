#!/usr/bin/env bash
# Check the remote run from anywhere. Usage: bash status.sh
# Run 3 (LCPO-Exact, 700 steps), launched 2026-08-24.
# The sshN.vast.ai proxy hangs on this box; use the direct ip:port.
HOST="${HOST:-80.188.223.202}"; PORT="${PORT:-11592}"; INST="${INST:-48542419}"
SSH="ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -p $PORT root@$HOST"

echo "=== training ==="
$SSH "tr '\r' '\n' < /workspace/go.log | grep -E '%\|' | tail -1; \
      grep -oE 'TRAIN_EXIT=[0-9]+' /workspace/go.log | tail -1; \
      ls -d /workspace/thinking-budget/checkpoints/lcpo-exact/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print \$2, \$0}' | sort -n | tail -1 | cut -d' ' -f2-" 2>/dev/null

echo "=== pipeline (tmux) ==="
$SSH "tmux ls 2>/dev/null; tail -3 /workspace/pipeline.log 2>/dev/null" 2>/dev/null

echo "=== dial check (mean_length should trend to ~1100, the mean budget; ~500 = run-2 collapse) ==="
$SSH "grep -oE \"'completions/mean_length': '[0-9.]+'\" /workspace/go.log | tail -3; \
      grep -oE \"'rewards/length_reward/mean': '[-0-9.]+'\" /workspace/go.log | tail -1; \
      grep -oE \"'kl': '[0-9.e-]+'\" /workspace/go.log | tail -1" 2>/dev/null

echo "=== gpu ==="
$SSH "nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader" 2>/dev/null

echo "=== spend ==="
.venv/bin/vastai show instance "$INST" --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
d=json.load(sys.stdin)
h=(d.get('duration') or 0)/3600
print('  %.2f hr x \$%.3f/hr = \$%.2f' % (h, d.get('dph_total',0), h*d.get('dph_total',0)))"
.venv/bin/vastai show user --raw 2>/dev/null | .venv/bin/python -c "
import json,sys; print('  credit left \$%.2f' % json.load(sys.stdin).get('credit',0))"

echo
echo "reattach pipeline:  $SSH -t 'tmux attach -t pipeline'"
echo "destroy when done:  .venv/bin/vastai destroy instance $INST"
