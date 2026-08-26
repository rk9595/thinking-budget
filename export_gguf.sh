#!/usr/bin/env bash
# Merge the LoRA into the base model, convert to GGUF, and quantize.
# Usage: bash export_gguf.sh checkpoints/lcpo-max/checkpoint-1000 out/l1-qwen-1.5b
set -euo pipefail

ADAPTER="${1:?usage: export_gguf.sh <adapter_dir> <out_dir>}"
OUT="${2:?usage: export_gguf.sh <adapter_dir> <out_dir>}"
BASE="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"

python - "$BASE" "$ADAPTER" "$OUT/merged" <<'PY'
import sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out = sys.argv[1:4]
m = AutoModelForCausalLM.from_pretrained(base, dtype="bfloat16")
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged ->", out)
PY

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$OUT/merged" --outfile "$OUT/model-f16.gguf" --outtype f16
for q in Q4_K_M Q8_0; do
  "$LLAMA_CPP/build/bin/llama-quantize" "$OUT/model-f16.gguf" "$OUT/model-$q.gguf" "$q"
done

echo "Done. Demo the budget knob:"
echo "  $LLAMA_CPP/build/bin/llama-cli -m $OUT/model-Q4_K_M.gguf \\"
echo "    -p 'What is 17*23?\\n\\nLet'\\''s think step by step and output the final answer within \\boxed{}. Think for maximum 200 tokens.'"
