# Verified-data SFT pilot: failed held-out gate

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

Both models completed all 300 generations on the separate 100-problem development set,
using budgets 512/1024/3600, seed 42, batch 32, and the same 8192-token ceiling and native template.
Generation finished on 2026-09-09; all raw outputs and checkpoints were retrieved locally.

The corrected report is [`regraded/comparison.json`](regraded/comparison.json).

| Budget | Base tokens | SFT tokens | Base accuracy | SFT accuracy | Base clipped | SFT clipped |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 5304 | 5604 | 46% | 33% | 36% | 57% |
| 1024 | 5329 | 6002 | 48% | 31% | 39% | 62% |
| 3600 | 5436 | 5894 | 46% | 33% | 39% | 61% |

SFT endpoint slope is 0.094 versus the base's 0.043 (ideal 1); only 31% of problems have
longer outputs at 3600 than at 512. Every budget's median SFT output hits the ceiling.
The gate declared in `ef83b8b` fails. **Do not initialize GRPO from this adapter or publish it.**

### Correctness audit on 2026-09-10

The original grader could credit an intermediate answer in unfinished reasoning when the
opening `<think>` was supplied by the prompt rather than generated. The fix uses the saved
rendered prompt to require the closing delimiter. It changes 9 base grades and 53 SFT grades.
Original raw evaluations remain in `../base-pilot-dev*` and `../sft-pilot-dev*`; audited copies
retain `legacy_correct`, the original generation manifests, and source hashes. No new
generation or threshold relaxation was used. The L1 reference still passes after the same audit.

The original reported SFT accuracies (51%/48%/51%) must not be interpreted as valid final-answer
accuracy. Corrected results show worse completion reliability as well as absent budget control.

## Next experiment, not yet authorized or run

Keep the failed adapter as evidence. Before another paid run, set a new budget and review the
cleanup incident below. Test a larger, more diverse verified SFT dataset while preserving the
existing 100-problem holdout. Audit termination and repeated/padded traces; compare budget-balanced
loss with the current token-weighted objective. Although example counts were equal, teacher
token shares were 10.1% / 19.8% / 70.0% across the budgets. Data scale and token weighting are
hypotheses to test, not proven causes of this pilot's failure. Reapply the held-out gate before GRPO.

## Rental cleanup incident

The unattended cleanup retrieved artifacts but omitted the CLI's `--yes` flag. A successful
command exit was incorrectly treated as deletion, and the spending watchdog was removed.
The instance therefore remained allocated. On 2026-09-10 it was explicitly destroyed with
`--yes`; two subsequent provider listings verified it absent. The account's available credit
is $0. No new paid instance or payment was initiated during the audit.

The provider's itemized charges for instance 50353651 total approximately **$4.079**, exceeding
the stated $3 limit: GPU $3.387, storage $0.661, download $0.028, upload $0.003. The earlier
balance-difference estimate of $3.68 understated these charges. Payment settlement is not inferred
from a zero available balance.

`cleanup_gpu.py` now passes the explicit confirmation flag and verifies the selected instance
is absent. An exited/stopped instance or command exit code 0 does not count as deletion. Tests
cover that failure mode. Keep spending safeguards active until provider-side deletion is verified;
local monitoring is not a provider-enforced billing cap.
