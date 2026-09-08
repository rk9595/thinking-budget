#!/usr/bin/env bash
# Evaluate one explicitly selected, fully saved checkpoint. Never upload or wait on pgrep.
# Usage: bash pipeline.sh <checkpoint-directory> <new-results-directory>
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
CHECKPOINT="${1:?usage: pipeline.sh <checkpoint-directory> <new-results-directory>}"
RUN_DIR="${2:?usage: pipeline.sh <checkpoint-directory> <new-results-directory>}"
PYTHON="${PYTHON:-python}"

"$PYTHON" - "$CHECKPOINT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve()
status = json.loads((p.parent / 'training_status.json').read_text())
if status['state'] not in ('paused', 'completed') or pathlib.Path(status['checkpoint']).resolve() != p:
    raise SystemExit('Checkpoint is not the explicitly saved paused/completed training checkpoint')
for name in ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json'):
    if not (p / name).is_file():
        raise SystemExit(f'Incomplete checkpoint: missing {name}')
PY

"$PYTHON" run_comparison.py --adapter "$CHECKPOINT" --out-dir "$RUN_DIR"
echo "Evaluation complete. Review comparison.json before any further training or release."
