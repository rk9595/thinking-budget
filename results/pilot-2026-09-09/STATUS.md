# Verified-data SFT pilot

## Data collection

- Frozen split: 500 training and 100 development problems, seed 29, with exact normalized
  MATH-500/GSM8K-test exclusions. No train/dev problem-ID overlap. This is not semantic decontamination.
- Teacher: L1-Qwen-1.5B-Exact at `b1fa57f192f0b14bd033d0085faaa80cdc39694b`.
- First draw: 500 problems × budgets 512/1024/3600, seed 42, batch 32 (1500 responses).
  Only 64 problems had correct, naturally terminated, closed-reasoning responses within 25%
  of all three budgets. The minimum-100 filter correctly stopped the first workflow before SFT.
- Second draw: the same 500 problems at 512/1024, seed 43, batch 32 (1000 more responses).
  The matching manifests and complete per-draw grids were validated before pooling.
- Pooled result: **102 complete problem sets / 306 examples**. All original quality thresholds
  were retained. Retry responses are training-data collection, not additional evaluation evidence.

First-draw diagnostics (out of 500 at each budget):

| Requested budget | Correct | Within 25% length | Correct + length + termination checks |
|---|---:|---:|---:|
| 512 | 199 | 336 | 130 |
| 1024 | 227 | 339 | 146 |
| 3600 | 285 | 499 | 285 |

The raw teacher traces and manifests remain under ignored `results/teacher-pilot*` locally;
the verified dataset and full teacher provenance are under ignored `data/pilot/`.

## Training

- Base: DeepSeek-R1-Distill-Qwen-1.5B at `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- One epoch, rank-32 all-linear LoRA, LR 1e-4, microbatch 1, accumulation 16, seed 42.
- Completed **20 optimizer steps** in 120.9 seconds on an A100 SXM4 40GB.
- Training loss 0.1971 is an implementation observation, not evidence of budget control.
- Full checkpoint, final adapter, configuration, source hashes, and dataset provenance are
  saved under ignored `checkpoints/sft-pilot/`. No weights have been uploaded or released.

The run emitted prompt-boundary warnings. An audit of all 306 examples found exactly one
boundary-token mismatch each: the prompt's newline combined with a teacher-leading newline
into a double-newline token. No reasoning text was lost from the loss mask. Commit `52dcaea`
removes the redundant newline and rejects unstable boundaries for future runs; all 306 examples
pass the real-tokenizer check after that change. This adapter was trained **before** that fix,
as its saved source hash records; it has not been silently relabeled as a rerun.

## Held-out evaluation

Matched base/student generation on the separate 100-problem development set is underway,
using budgets 512/1024/3600, seed 42, batch 32, and the same 8192-token ceiling and native template.
The pilot gate was declared in commit `ef83b8b` before training. Do not start GRPO or treat this
as a successful budget model until the complete paired report and raw traces have been reviewed.
