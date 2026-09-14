# Reference-control experiment: passed on 2026-09-09

Correctness was audited on 2026-09-10 without new generation. Use
[`../control-2026-09-10-regraded/comparison.json`](../control-2026-09-10-regraded/comparison.json)
for corrected accuracy: base-Exact at budget 3600 is 84%, not 86%. The reference still passes.
The original generation results below are retained for provenance.

The 50 MATH-500 development problems and three model revisions are frozen in this directory.
There are four arms because the L1 reference uses different wording and a different chat template;
each trained model has a matching base-model control. All 600 responses are saved with their raw
text, token IDs, stop reasons, and manifests. `comparison.json` can be regenerated from these files.

## Results

Each cell reports mean generated tokens / accuracy on the same 50 problems, seed 42.

| Arm | Requested 512 | Requested 1024 | Requested 3600 |
|---|---:|---:|---:|
| Base, Exact prompt | 3291 / 80% | 3015 / 84% | 3241 / 86% |
| Run 3, identical prompt | 923 / 70% | 887 / 68% | 962 / 66% |
| Base, L1 prompt/template | 3635 / 80% | 3546 / 86% | 3501 / 78% |
| L1 reference, identical prompt/template | 512 / 76% | 924 / 78% | 3578 / 88% |

The reference passes all six predeclared development checks. Its within-problem endpoint slope
is 0.993 (ideal 1), versus −0.044 for its matched base. All 50 problems have longer outputs at
3600 than at 512; 49/50 increase across all three budgets. Mean absolute relative errors are
26.2%, 16.0%, and 6.7%; a mean of 512 at the smallest budget does not imply exact individual adherence.
Reference truncation rates are 0%, 0%, and 2% under the common 8192-token completion ceiling.

Run 3 remains a compression model, not a functioning dial: endpoint slope 0.013. Its accuracy
differences from the matched base are −10, −16, and −20 percentage points. Problem-bootstrap 95%
intervals are [−24, +4], [−30, −4], and [−34, −6] points. This small, single-seed development
comparison has different prompts, budgets, problems, and generation settings from the historical
run-3 report; it does not replace a full benchmark or establish the cause of the discrepancy.

This passes the evaluator/reference gate, **not** a trained-student or release gate. The next
stage is verified teacher generation on the separate training split, followed by an SFT pilot.

## Execution

- Source: `fd296a8`; frozen batch size 32, model revisions and problem hash in `comparison_plan.json`.
- GPU: one A100 SXM4 40GB; Python 3.12.13, torch 2.13.0+cu130, vLLM 0.27.1.
- CUDA lock installed; dependency check and 24 unit tests passed on the GPU machine.
- Two-problem CUDA wiring check passed separately in `../cuda-smoke-2026-09-09/`.
- Full comparison finished with exit 0 at approximately 06:58 UTC; artifacts recovered locally.

Local verification on 2026-09-08:

- Reward and evaluation unit checks pass, including rejecting reversed/constant budget responses,
  detecting incomplete sample grids, preserving full SFT reasoning, and rejecting truncated teacher examples.
- A real tiny random model exercised the TRL training loop on CPU: pause at step 1, reject a changed
  learning rate, resume optimizer/scheduler state, and finish step 2. This checks implementation wiring,
  not learned budget control.
- The local dependency lock installs and passes `uv pip check`.
- The Linux CUDA dependency lock was subsequently installed and executed successfully on 2026-09-09.
- Pilot splits were prepared locally: 500 training and 100 separate development problems, with no
  overlapping IDs. A separate GRPO dataset contains 1,500 rows, 500 at each of 512/1024/3600 tokens.

Local regression suite rerun on 2026-09-09: 25 tests passed, including the real CPU pause/resume test.
