# thinking-budget

Train a small reasoning model to obey an explicit thinking-token budget ("Think for maximum N tokens") using GRPO. Reproduces the L1/LCPO recipe (Aggarwal & Welleck, CMU 2025) on DeepSeek-R1-Distill-Qwen-1.5B, then extends it with a GGUF release and a llama.cpp demo of the budget knob.

Reward: `r = 1[answer correct] − α·|N − tokens_used|` (LCPO-Exact, α=3e-4), correctness via `math-verify` on the boxed answer. The budget is a **target**: both overshoot and undershoot are penalized, which is what forces the model to condition on N instead of just going short. `--length-reward max` switches to the LCPO-Max variant (`−α·max(0, tokens_used − N)`), which treats the budget as a ceiling only.

The prompt wording must match the reward variant ("exactly N" for Exact, "maximum N" for Max). It is set in one place — `TB_LENGTH_REWARD`, read by `rewards.budget_instruction()` — because `prepare_data.py`, `eval_budget.py` and `profile_inference.py` build prompts in separate processes and run 1 was lost to them disagreeing. `train_grpo.py` reads the wording back off `data/train` and refuses to start on a mismatch.

## Run history

**2026-08-24/25 (LCPO-Exact, 600 steps, 8.4h) — run 3: no dial, but a real compression result.** Exact penalizes undershoot as well as overshoot, which should force conditioning on N in both directions. It did not. MATH-500 mean tokens 781/783/763/821 across budgets 256/512/1024/2048 — a **1.08x spread**. The sweep shows tokens compressing (2178 → 965 → 776 → 751 at steps 100/250/500/600) while the spread never exceeds 1.17x: the model regressed to the **conditional mean** of the budget distribution rather than learning to read N.

What it did buy is efficiency: **MATH-500 2524 → 787 tokens (3.2x) for −2.0 accuracy points**, GSM8K 1239 → 457 (2.7x) for −5.2. Run 2 paid 8–13 points for less.

The diagnostic is `kl` = **0.006**, five times *lower* than run 2's 0.03 at an identical LR — Exact's gradients partially cancel (high budgets say "write more", low say "write less"; they reconcile only if the policy attends to N). The optimizer was healthy: `frac_reward_zero_std` 0, `clipped_ratio` 0. Final `rewards/length_reward/mean` −0.199 ⇒ mean |n−N| ≈ 660 tokens. **The untested lever is `ALPHA`:** at 3e-4 the length penalty (0.199) is ~half the correctness std (0.44), so length is a minority of the gradient. Try α≈1e-3, a higher LoRA rank, or a full fine-tune. Archived at `results/run-2026-08-24-exact/`; curves in W&B run `vt32er6f`.

**2026-08-23 (LCPO-Max, 700 steps, 7.6h, $8.12) — partial success.** All three run-1 fixes worked: `kl` 6e-4 → 0.03, mean completion length 4096 → 670, `clipped_ratio` 0.97 → 0. Ceiling compliance is real — MATH-500 over-budget rate 87% → 3% at budget 1024, 53% → 0% at 2048. **But mean tokens is flat across budgets** (490/559/546/528 for 256/512/1024/2048): the model learned "always be short", not "condition on N", which is the degenerate solution of a max-only reward — nothing penalizes finishing early, so nothing pulls length *up* toward the budget. Collapse was complete by step 100. Accuracy cost: MATH-500 −8 to −13 pts, GSM8K −1 to −5. Training-time correctness did not reveal this (0.298 → 0.277, noise-dominated at 32 samples/step); only held-out eval did. Archived under `results/run-2026-08-23-max/` (code snapshot in `code/`).

**2026-08-22 (LCPO-Exact, 1000 steps, $24.75) — negative result.** Mean tokens stayed flat across budgets (2190/2197/2225/2229 on MATH-500 for budgets 512/1024/2048/3600); accuracy was unharmed. Archived under `results/run-2026-08-22-exact/`, adapter at [rk9595/thinking-budget-qwen1.5b-lcpo](https://huggingface.co/rk9595/thinking-budget-qwen1.5b-lcpo). Three causes, all fixed on `main`:

1. **The policy never moved.** `grad_norm` ~0.004 and `kl` ~6e-4 end to end at `lr=5e-6`. That is a full-finetune LR applied to a LoRA adapter; LoRA needs roughly 10x more. Now `2e-5`.
2. **Prompt and reward disagreed.** The prompt said "Think for **maximum** N tokens" while the reward was LCPO-**Exact** (`−α·|N − n|`), which pays the model to pad up to N. Now LCPO-Max by default, matching the prompt.
3. **A third of the data pushed the wrong way.** Budgets were sampled 100–3600 against a natural length of ~2500, so high-budget examples rewarded writing *longer*. Now 200–2000, entirely below the natural length, with the 4096 completion cap left as headroom.

## Known environment issue (vLLM 0.27 + Python 3.11)

Export `VLLM_USE_FLASHINFER_SAMPLER=0` and `pip uninstall -y flashinfer-python` before any
vLLM run. flashinfer 0.6.x fails to import on Python 3.11 (`array.array is not
subscriptable`), but vLLM's `flashinfer_sampler_supported()` imports the backend
unguarded, so simply removing the package trades one crash for a `ModuleNotFoundError`.
The env var short-circuits before that import; both are needed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm          # GPU box only
```

## Workflow

```bash
python -m pytest tests/ -q                     # 11 reward unit tests
python prepare_data.py                         # DeepScaleR subset + budget injection -> data/train
python prepare_data.py --num-samples 64 --out data/smoke
python train_grpo.py --smoke --data data/smoke # 2-step Qwen3-0.6B run to validate the loop
python eval_budget.py --out results/base.json  # baseline: base model ignores budgets
python train_grpo.py --output-dir checkpoints/lcpo-exact --max-steps 1000   # main run (A100)
python eval_budget.py --lora checkpoints/lcpo-exact/checkpoint-1000 --out results/trained.json
python plot_results.py                         # curves + markdown table for the model card
bash export_gguf.sh checkpoints/lcpo-exact/checkpoint-1000 out/l1-qwen-1.5b
```

On a rented GPU box, `bash setup_gpu.sh` does the install, runs the tests, and builds the
dataset in one shot. It requires `HF_TOKEN` and `WANDB_API_KEY` in the environment.

To do the whole thing from here — provision, train, fetch results, destroy the box:

```bash
export VAST_API_KEY=... HF_TOKEN=... WANDB_API_KEY=...
bash rent_and_run.sh          # refuses offers above MAX_DPH (default $1.30/hr)
```

It auto-destroys the instance after pulling results down; pass `--keep` to leave it running.
Needs a vast.ai account with credit loaded — the API key alone is not enough.

Resume an interrupted run with `--resume`. Training logs to W&B; watch `rewards/correctness_reward/mean` (should not decrease) and `rewards/length_reward/mean` (should rise toward 0 as the model learns to hit the budget). Note TRL names that metric after the reward function's `__name__`, so it is `length_reward_max` for LCPO-Max and `length_reward` for LCPO-Exact — a grep written for one silently matches nothing on the other. Also watch `frac_reward_zero_std`: if it stays near 1, every completion in a group scores identically, advantages are zero and GRPO learns nothing. And watch `kl` — if it is still ~1e-4 after a few hundred steps the policy is frozen and the run is already dead, whatever the reward curve looks like.

`--smoke` runs on CPU — the trainer's generation path hits `probability tensor contains inf/nan` on Apple MPS (plain `model.generate` on MPS is fine, so this is inside Trainer/accelerate, not the reward code). It takes ~60s/step on an M-series Mac and completions clip at 64 tokens, which is expected; it validates wiring only, not learning.

## Eval

`eval_budget.py` reports, per dataset × budget: accuracy, mean tokens used, `overshoot_rate` (fraction of problems exceeding the budget) and `mean_overshoot` (mean `max(0, tokens − budget)`). `mean_abs_deviation` is still logged so runs stay comparable to the LCPO-Exact archive. Datasets: MATH-500, GSM8K, AIME24; budgets {256, 512, 1024, 2048} — 3600 was dropped because it sits above the base model's natural ~2500 tokens, so nothing ever overshoots it.

Success for LCPO-Exact = `mean_tokens` rises monotonically with the budget and tracks the y=x line in `tokens_vs_budget.png` (this is the headline plot — it is the one thing run 2 failed), `mean_abs_deviation` well below base at every budget, and accuracy rising with budget. Accuracy at the top budget matching the base model would be a bonus, not a pass condition: this is LoRA r=32, not the full fine-tune the L1 paper reports.

For LCPO-Max the criterion is different and weaker — `overshoot_rate` falls sharply at every budget — because a model that is uniformly short satisfies it without learning anything.
