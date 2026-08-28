---
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
library_name: peft
license: mit
tags:
  - grpo
  - reasoning
  - test-time-compute
datasets:
  - agentica-org/DeepScaleR-Preview-Dataset
---

# Thinking-Budget Qwen-1.5B (LCPO-Exact)

A LoRA adapter for DeepSeek-R1-Distill-Qwen-1.5B, trained with GRPO on the LCPO-Exact
reward from [L1](https://arxiv.org/abs/2503.04697) (Aggarwal & Welleck, 2025). Code and all
three runs: [github.com/rk9595/thinking-budget](https://github.com/rk9595/thinking-budget).
Writeup of all three runs, including the two that failed and why the conditioning never
formed: [Three GRPO runs to give a 1.5B model a thinking-budget
dial](https://rakeshkariya.vercel.app/blog/thinking-budget-lcpo).

**What it actually does: it reasons ~3.2x more concisely than the base model for about 2
points of accuracy on MATH-500.** That is the result. It is a fixed compression, not a
control knob.

**What it does not do: respond to the budget you ask for.** Despite being trained on
`Think for exactly N tokens`, its output length barely moves with N — 781/783/763/821 mean
tokens for budgets 256/512/1024/2048, a 1.08x spread. It learned to be uniformly short
rather than to condition on N. Do not use this expecting a test-time-compute dial. The
budget number in the prompt is close to inert; what you get is the compression.

LoRA r=32, 600 GRPO steps on an A100 80GB (~8.4 h).

## Reward

`r = 1[answer correct] − α·|N − tokens_used|`, α = 3e-4, correctness via `math-verify` on
the boxed answer. Unlike the Max variant, undershoot is penalized too — the budget is a
target, not a ceiling. That was intended to force conditioning on N; see
[Run history](#run-history) for why it did not.

## Results

100 problems per dataset × budget, temperature 0.6, top-p 0.95, vs. the unmodified base model.

| dataset | budget | tokens base | tokens trained | compression | acc base | acc trained |
|---|---|---|---|---|---|---|
| math500 | 256 | 2566 | **781** | 3.28x | 0.71 | 0.69 |
| math500 | 512 | 2540 | **783** | 3.24x | 0.75 | 0.73 |
| math500 | 1024 | 2476 | **763** | 3.25x | 0.78 | 0.77 |
| math500 | 2048 | 2516 | **821** | 3.06x | 0.77 | 0.74 |
| gsm8k | 256 | 1360 | **447** | 3.04x | 0.85 | 0.78 |
| gsm8k | 512 | 1087 | **469** | 2.32x | 0.85 | 0.78 |
| gsm8k | 1024 | 1308 | **444** | 2.95x | 0.82 | 0.80 |
| gsm8k | 2048 | 1203 | **468** | 2.57x | 0.85 | 0.80 |

Averaged: MATH-500 **2524 → 787 tokens (3.2x) for −2.0 accuracy points**; GSM8K
**1239 → 457 (2.7x) for −5.2 points**.

**Read the tokens column down, not across.** Within a dataset it is nearly constant. That
constancy *is* the negative result.

Budget adherence is therefore lopsided. Mean `|tokens − budget|` on MATH-500 improves
sharply at tight budgets (2310 → 525 at 256; 2030 → 341 at 512) and gets *worse* at loose
ones (1149 → 1278 at 2048), because a model that always writes ~790 tokens undershoots 2048
as badly as the base model overshoots it.

## Limitations

- **Not a compute dial.** Output length does not track the requested budget, and accuracy
  does not rise with it. If you need genuine budget control, this adapter does not provide it.
- Accuracy is at or slightly below base at every budget tested. Use it when you want shorter
  reasoning, not better reasoning.
- GSM8K pays more (−5.2 pts) than MATH-500 (−2.0). Its natural completions are already
  short, so the same compression cuts closer to the bone.
- Trained on competition math (DeepScaleR); behaviour on out-of-domain prompts is untested.
- LoRA r=32, not a full fine-tune. L1's published numbers come from a full fine-tune and
  this does not match them.
- No GGUF quants in this repo yet.

## Usage

The prompt wording must be `exactly`, matching training. (An earlier version of this card
said `maximum`; that was the wording for a different, superseded run.)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base),
    "rk9595/thinking-budget-qwen1.5b-lcpo",
    subfolder="adapter",
)
tok = AutoTokenizer.from_pretrained(base)

prompt = ("What is 17*23?\n\nLet's think step by step and output the final answer "
          "within \\boxed{}. Think for exactly 200 tokens.")
```

## Run history

Three runs, each isolating a different cause. All are archived in the code repo.

**Run 1 (2026-08-22, LCPO-Exact, 1000 steps).** No length control at all; mean tokens flat
at ~2200 regardless of budget. Three independent bugs: the LoRA learning rate was 5e-6, a
full-finetune value, so the policy never moved (`kl` 6e-4 after 1000 steps); the prompt said
`maximum N` while the reward was Exact, which pays the model to pad *up* to N; and budgets
were sampled 100–3600 against a natural length of ~2500, so a third of the data rewarded
writing longer.

**Run 2 (2026-08-23, LCPO-Max, 700 steps).** With those fixed (lr 2e-5, budgets 200–2000),
the ceiling was respected — over-budget rate on MATH-500 fell 87% → 3% at budget 1024 — but
mean tokens was flat at ~530 and accuracy fell 8–13 points. This is the degenerate solution
of a max-only reward: nothing penalizes finishing early, so nothing pulls length up toward
the budget. Collapse was complete by step 100.

**Run 3 (2026-08-24, LCPO-Exact, 600 steps) — this model.** Exact penalizes undershoot too,
which should force conditioning on N in both directions. It did not. Tokens compressed
steadily (2178 → 965 → 776 → 751 at steps 100/250/500/600) while the spread across budgets
never exceeded 1.17x: the model regressed to the *conditional mean* of the budget
distribution instead of learning to read N.

The training curve says the run was **truncated, not converged**. Binned over 50 steps, `kl`
was still climbing at the end (0.0106 across steps 501–600) and mean completion length was
still falling (3638 → 1375 tokens); `rewards/length_reward/mean` improved monotonically
−0.747 → −0.242. Run 2 by contrast plateaued at `kl` ≈ 0.032 and ~700 tokens from step 300
onward. So Exact is not stalled — it moves the same direction as Max but roughly 3x slower
per step, which is what partial gradient cancellation looks like: high budgets say "write
more", low budgets say "write less", and the two reconcile only if the policy attends to N.

What shows *no* trend is conditioning itself. Across the checkpoint sweep the MATH-500
budget spread went 1.065 → 1.164 → 1.103 → 1.170 at steps 100/250/500/600 while mean length
fell 3x. Compression improved steadily; conditioning never started, so more of the same
training would likely have bought more compression and no dial. The optimizer was healthy
throughout (`frac_reward_zero_std` = 0, so advantages were real; `clipped_ratio` = 0, so
nothing was truncated), and entropy fell 0.75 → 0.35 — the policy collapsing onto a single
length *habit* rather than learning a length *function*. Final
`rewards/length_reward/mean` = −0.199, i.e. mean `|n − N|` ≈ 660 tokens.

**The most likely fix, untested here, is α.** At 3e-4 the length penalty (0.199) is roughly
half the correctness reward's standard deviation (0.44), so length is a minority of the
gradient signal. α ≈ 1e-3, a higher LoRA rank, or a full fine-tune (what L1 used) are the
levers worth trying next. Simply training longer is *not* one of them: the conditioning
signal was flat for 600 steps, so extra steps buy compression, not a dial.

## Reproduce

The exact scripts this checkpoint was trained with are snapshotted in [`code/`](./code) of
this repo — `prepare_data.py`, `rewards.py`, `train_grpo.py`, `eval_budget.py`, `pipeline.sh`.
The full project, including the run-1 and run-2 archives, is at
[github.com/rk9595/thinking-budget](https://github.com/rk9595/thinking-budget); the writeup
is [here](https://rakeshkariya.vercel.app/blog/thinking-budget-lcpo).

```bash
TB_LENGTH_REWARD=exact python prepare_data.py --out data/train   # 8k DeepScaleR, budgets 200-2000
python train_grpo.py --length-reward exact --max-steps 600 \
    --lr 2e-5 --micro-batch 2 --vllm-mem 0.20
python eval_budget.py --lora checkpoints/lcpo-exact/checkpoint-600 \
    --budgets 256,512,1024,2048 --n 100
```

One A100 80GB, ~8.4 h for 600 steps (45.8 GB peak, 98% utilization), 44.9M completion
tokens. `results/` holds the base and trained evals plus the per-checkpoint sweep.
