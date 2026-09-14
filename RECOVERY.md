# Recovery workflow

The first milestone is a reliable three-model comparison. Training and publishing are separate actions.
The old Run 4 range-only experiment is superseded. This workflow tests an alternative SFT → GRPO
recipe; it is not a faithful reproduction of the original full-finetuning experiment.

## Current status

- CPU reward/evaluation checks and a real tiny-model GRPO pause/resume integration test are provided.
- The 50-problem four-arm comparison **passed** on an A100 on 2026-09-09. Raw responses,
  manifests, paired statistics, and interpretation are in
  [`results/control-2026-09-10-regraded/STATUS.md`](results/control-2026-09-10-regraded/STATUS.md).
  L1 averages 512/924/3578 tokens; run 3 stays at 923/887/962 for targets 512/1024/3600.
- `prepare_pilot.py split` prepares 500 training and 100 separate development problems locally.
  Generated data lives under ignored `data/pilot/`; rerun the command on the GPU machine or copy it.
- Dependency locks resolve for Python 3.12. `requirements.txt` is the local macOS lock;
  `requirements-gpu.txt` is the Linux x86-64 CUDA lock. Both installed successfully; the CUDA
  environment passed its dependency check and 24 unit tests. The expanded local suite includes
  grading, non-destructive regrading, and provider-side cleanup verification regressions.
- Two teacher draws yielded 102 verified three-budget sets (306 examples). SFT completed 20
  optimizer steps but **failed** the full 100-problem held-out evaluation: mean lengths
  5604/6002/5894, truncation 57%/62%/61%, corrected accuracy 33%/31%/33%.
  [Pilot report and next experiment](results/pilot-2026-09-09/STATUS.md).
- The grading audit rejects unfinished reasoning even when `<think>` comes from the prompt.
  Original raw outputs are preserved alongside corrected copies. The reference still passes.
- The temporary instance is verified deleted. A cleanup error exceeded the stated $3 limit:
  provider charges total approximately $4.08. See the pilot report
  for the incident and fix.
- On September 13, the user authorized a staged $10 recovery budget: up to $3 for the
  small SFT learnability diagnostic, up to $5 for a larger verified-data SFT pilot only if
  that diagnostic passes, and $2 reserved for overhead and cleanup. This does not authorize
  GRPO, publication, or additional spending. The frozen diagnostic plan is
  [`results/sanity-2026-09-13/plan.json`](results/sanity-2026-09-13/plan.json).
  It compares token-weighted and equal-completion NLL from identical initial seeds on 24
  training problems for 20 epochs. Twelve other verified-source problems are diagnostic
  probes; the original 100-problem holdout is untouched. Passing training-set controls and
  75% accuracy at each budget permits the larger pilot, not a release claim.
  `paid_sanity.py` retrieves and hashes artifacts before verified destruction. Its separate
  remote timeout stops compute using the instance-restricted key, but stopping alone does
  not end storage charges. Provisioning has a 15-minute local timeout; job cleanup begins
  at 80 minutes and remote compute stop is scheduled at 90 minutes from creation.
- That diagnostic completed: both variants learned a strong within-problem budget response,
  but both failed the per-response length-error gate. The larger pilot was not started.
  Both adapters and all 324 outputs were backed up and checksum-verified before GPU deletion.
  Reported charges total $0.519. See the
  [completed diagnostic and targeted next step](results/sanity-2026-09-13/STATUS.md).
- On September 14, the user requested continuation and a viability assessment. The next
  [frozen decoding-only diagnostic](results/decode-2026-09-14/PLAN.md) reuses the saved
  adapters at two fresh sampling seeds plus greedy decoding. Its additional limit is $1.50
  within the original diagnostic allocation. It cannot automatically unlock stage two.

## 1. Install and validate

On the Mac:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
```

On a Linux CUDA machine with Python 3.12:

```bash
bash setup_gpu.sh
source .venv/bin/activate
```

Do not run `rent_and_run.sh`: it is disabled because its legacy teardown can destroy a run
on SSH disconnection. Provisioning and an explicit total spending limit must be settled before
starting any paid job. Run long commands under tmux and copy artifacts before releasing the machine.
After verifying backup completeness and checksums, use:

```bash
python cleanup_gpu.py --instance INSTANCE_ID --backup-dir /absolute/path/to/recovered/artifacts
```

This explicitly confirms deletion and checks that the selected instance is absent from the provider.
Keep the watchdog active if verification fails. A stopped instance still incurs storage charges;
a successful CLI exit is not proof of deletion. Local watchdogs are not provider-enforced billing caps.

## 2. Run the comparison

```bash
python run_comparison.py --out-dir results/control-next --prepare-only
python run_comparison.py --out-dir results/control-next
```

This evaluates three models in four arms, on the same 50 problems at 512/1024/3600 tokens:

| Arm | Model | Prompt and template |
|---|---|---|
| base_exact | DeepSeek R1 Distill Qwen 1.5B | exactly N; native DeepSeek template |
| run3_exact | Same base + frozen published adapter | identical to base_exact |
| base_l1 | Same base without adapter | original L1 wording and L1 template |
| reference_l1 | Released L1-Exact | identical to base_l1 |

The extra base arm controls both wording and the different `<think>` prefill behavior.
No budget-dependent hard cap or forced stopping is used: all budgets have 8192 completion-token headroom.
The comparison uses batches of 32 on the A100; this setting is frozen in the plan and each manifest.
The primary length measure is **total generated token IDs**, including EOS if the backend returns it.
Reasoning and answer token counts are also recorded; missing `</think>` yields unknown reasoning length,
not zero. This experiment does not claim to measure a reasoning-only budget.

Each arm writes a manifest, `.samples.jsonl` with raw responses/token IDs/stop reasons, and an aggregate JSON.
`comparison.json` adds matched accuracy/token differences and problem-level bootstrap confidence intervals.
The fixed pilot gate checks reference length error, truncation, accuracy, and positive within-problem
response to budgets relative to the base. A nonzero exit from the comparison means inspect the report;
it must not automatically trigger training or publication. This small development gate is not a release gate.

Completed arms may be reused when rerunning the same frozen plan. Partial arms retain raw evidence but
currently require a new output directory to rerun; they are never silently overwritten.
Plans using the older grader cannot be silently reused by the current runner. Existing generation
can instead be audited locally with `regrade_eval.py --input OLD.json --out NEW.json`, then compared
with `compare_results.py` or `compare_pilot.py`. Do not mix grading versions between paired arms.

For an initial CUDA wiring check, use a separate plan with `--limit 2`; do not mistake this for the
50-problem comparison. Keep the official model and tokenizer revisions frozen by the plan.

## 3. Prepare verified SFT examples

Only after the reference comparison passes:

```bash
python prepare_pilot.py split --out-dir data/pilot --train-count 500 --dev-count 100
python eval_budget.py --model l3lab/L1-Qwen-1.5B-Exact \
  --revision b1fa57f192f0b14bd033d0085faaa80cdc39694b \
  --wording l1 --problem-file data/pilot/train_problems.jsonl \
  --budgets 512 1024 3600 --batch-size 32 --out results/teacher-pilot.json
python prepare_pilot.py filter --teacher-eval results/teacher-pilot.json \
  --train-problems data/pilot/train_problems.jsonl --out data/pilot/sft.jsonl
```

Skip `split` if copying the already prepared directory. It refuses to overwrite an existing split.
Splitting deduplicates questions before assigning train/dev and excludes normalized exact matches against
MATH-500 and GSM8K test. This is not semantic decontamination. Keep all variants of a question in one split.
Filtering rechecks answer correctness, requires natural termination and a closed reasoning trace,
and retains complete sets of three budgets within 25% of their targets. If fewer than 100 complete
problem sets survive, inspect the data; do not silently lower the bar or launch a full job.

The first 500-problem draw on 2026-09-09 produced only 64 complete accepted sets. A second
draw at seed 43 was therefore requested at the bottleneck budgets 512 and 1024; the original
3600-token samples are reused. Multiple draws can be pooled without relaxing the quality filter:

```bash
python eval_budget.py --model l3lab/L1-Qwen-1.5B-Exact \
  --revision b1fa57f192f0b14bd033d0085faaa80cdc39694b \
  --wording l1 --problem-file data/pilot/train_problems.jsonl \
  --budgets 512 1024 --seeds 43 --batch-size 32 --out results/teacher-pilot-retry.json
python prepare_pilot.py filter --teacher-eval results/teacher-pilot.json results/teacher-pilot-retry.json \
  --train-problems data/pilot/train_problems.jsonl --out data/pilot/sft.jsonl
```

Each draw must cover its complete declared grid and use the same teacher revision, template,
problem set, and sampling configuration apart from budgets/seeds. The SFT manifest preserves
every source manifest. Training examples remain one verified response per problem/budget;
retries are data collection, not additional independent evaluation evidence.
The generated teacher traces, operational logs, filtered dataset, and checkpoint remain local
artifacts (ignored by git); they are not a published dataset/model. Preserve them before ending
the rental. Public result reports do not substitute for those training artifacts.

```bash
python train_sft.py --data data/pilot/sft.jsonl \
  --control-report results/control-2026-09-10-regraded/comparison.json \
  --out-dir checkpoints/sft-pilot
python eval_budget.py --problem-file data/pilot/dev_problems.jsonl \
  --budgets 512 1024 3600 --batch-size 32 --out results/base-pilot-dev.json
python eval_budget.py --lora checkpoints/sft-pilot/final \
  --problem-file data/pilot/dev_problems.jsonl --budgets 512 1024 3600 \
  --batch-size 32 --out results/sft-pilot-dev.json
python compare_pilot.py --base results/base-pilot-dev.json \
  --candidate results/sft-pilot-dev.json --out results/sft-pilot-comparison.json
```

SFT preserves raw reasoning explicitly: these models' completed-message chat templates otherwise strip CoT.
Loss applies only to the completion. Oversize examples fail rather than silently losing the final answer.
The pilot has one epoch, rank 32, LR 1e-4; these are proposed starting settings, not proven successful settings.
Inspect paired budget response, accuracy, truncation, and repeated/padded text before continuing.
`compare_pilot.py` requires matched model revisions, templates, samples, and generation settings.
Its development gate reuses the reference-control checks and additionally requires high-budget
accuracy to lose no more than 10 percentage points versus the matched base. These point-estimate
thresholds screen a small pilot for GRPO, not a release; read the paired bootstrap intervals too.
The thresholds were declared before the first SFT pilot's training and held-out generation.

## 4. Conditional GRPO refinement

Only proceed if the held-out SFT pilot actually responds to budgets. Prepare GRPO data separately;
do not reuse evaluation problems. `--init-adapter checkpoints/sft-pilot/final` starts from the pilot.
The trainer now exposes response batch size, number of generations, rank, alpha, seed, sampling,
loss type, normalization, and completion cap. Save and inspect `run_manifest.json` before scaling.

To pause while retaining the intended learning-rate schedule, specify the full planned step count
and an earlier stop boundary:

```bash
TB_LENGTH_REWARD=exact python prepare_data.py \
  --problem-file data/pilot/train_problems.jsonl --budgets 512 1024 3600 --out data/train
python train_grpo.py --data data/train --output-dir checkpoints/refinement \
  --init-adapter checkpoints/sft-pilot/final --max-steps 700 --stop-after-step 100
bash pipeline.sh checkpoints/refinement/checkpoint-100 results/refinement-100
```

`pipeline.sh` validates the explicitly saved checkpoint, evaluates it, and never publishes.
Its `run3_exact` arm name is a legacy label for the selected adapter; the manifest records the actual path/hash.
To resume, repeat the original training options with `--resume`, removing or advancing `--stop-after-step`.
The trainer rejects changed configurations/datasets and reports `paused`, `completed`, or `failed` explicitly.
The learning-rate schedule continues from the saved optimizer/scheduler state.

Do not infer a working dial from KL, average reward, or compression alone. Inspect within-group
length variation and held-out response to budgets. Four question/budget groups per update remains
the default for comparability; raising the sampling budget is an explicit, separately measured decision.

## 5. Final validation and release (pending)

Use additional sampling seeds and full benchmark reporting; distinguish the 50 development problems
from untouched problems and bootstrap by problem. Proposed engineering targets: mean lengths within
20% of requested budgets, positive paired budget response, and high-budget accuracy within 3 percentage
points of the chosen base, reporting uncertainty rather than declaring statistical equivalence.

After reviewing a successful model, merge/export GGUF and rerun accuracy/adherence with the identical
rendered prompts. Inspect quantization regressions before publishing. Publish a new version/repository
with the matching model card and data lineage; preserve run 3 as the existing compression artifact.

There is deliberately no automatic promotion from training completion or a passed pilot gate to a public model.
