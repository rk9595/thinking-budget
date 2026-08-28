# Run 4 — the experiment that was never run

**Status: not started. Blocked on vast.ai credit ($0.00 as of 2026-08-28).**

Runs 1–3 produced a 3.2x compression and no budget dial. This is the plan for the one
configuration that has not been tried, why it is the right next variable, and what it costs.

## The hypothesis

L1 samples training budgets from `U(100, 4000)`, straddling the model's natural ~2500-token
length. Run 3 sampled `U(200, 2000)`, entirely below it.

A policy that ignores `N` and emits a constant pays the mean absolute deviation of the budget
distribution. For `U(a,b)` the best constant is the median and the cost is `(b−a)/4`:

| range | cost of ignoring N |
|---|---|
| run 3, `U(200, 2000)` | **450 tokens** |
| L1, `U(100, 4000)` | **975 tokens** |

So refusing to condition was **2.2x cheaper here than in the paper**. At α = 3e-4 that is a
0.135 penalty against a correctness spread of 0.44 — a minority shareholder in the gradient.
Under L1's range the same constant strategy costs 0.293, the same order as correctness.

The qualitative half matters more than the arithmetic. With every budget below the natural
length, **"just be shorter" satisfies every training example at once.** All gradients point
one way, so a uniform length habit is not an approximation — it is a complete solution.
Straddling is what makes `N` informative in both directions: low budgets say shorten, high
budgets say lengthen, and the only policy satisfying both is one that reads `N`.

Since the penalty for ignoring `N` is `α · (b−a)/4`, widening the range and raising α are the
same knob. Widening is free, and it is what the paper did.

## Why this ordering

Run 1 sampled 100–3600 — straddling, essentially L1's setting. It failed for unrelated
reasons (lr 10x too small for LoRA, prompt/reward wording mismatch), and the range was
written up as the third bug and "fixed" to 200–2000. The two real bugs were fixed; the range
change was never independently justified.

**Run 1 had the right range and broken everything else. Runs 2 and 3 had the right everything
else and the wrong range. The correct combination has never been run.**

## Spec

Change exactly one thing from run 3:

| | run 3 | run 4 |
|---|---|---|
| budget range | `U(200, 2000)` | **`U(200, 3600)`** |
| length reward | Exact, α=3e-4 | unchanged |
| LoRA | r=32, α=64, lr 2e-5 | unchanged |
| steps | 600 | 700 |
| base model | DeepSeek-R1-Distill-Qwen-1.5B | unchanged |
| `max_completion_length` | 4096 | unchanged |

**Why 3600 and not L1's 4000.** `max_completion_length` is 4096. At N=4000 there is ~96
tokens of headroom, so overshoot is truncated and the penalty is clipped at exactly the
budgets where conditioning has to show up. Raising the cap costs memory and step time for no
extra signal. 3600 straddles the natural ~2500 just as well and puts the cost of ignoring N
at 850 tokens — 1.9x run 3, within reach of L1's 975.

## Cost

Wider budgets mean longer mean completions, so steps run slower than run 3's 71/hr. Expect
60–70/hr.

| | |
|---|---|
| training, 700 steps | 10–12 h |
| pipeline (eval + sweep + profile + upload) | ~2 h |
| at ~$1.09/hr | **$14–16** |
| **top up** | **$25** — margin so a slow box does not strand the run |

Needs an A100 80GB at ≤$1.30/hr, and rotated `HF_TOKEN` + `WANDB_API_KEY` in
`~/.thinking-budget.env`.

## Steps

`pipeline.sh` already evaluates at `256 512 1024 2048 3600` (committed with this plan) — a
dial would first appear in the region above the old 2048 ceiling, so evaluating only to 2048
could not see it. The latency profile still uses the old four budgets on purpose; it measures
throughput, not adherence.

```bash
# 1. rsync code up, excluding .venv, checkpoints, data, results.
#    Credentials go via stdin into /root/.env - never on an ssh command line.

# 2. install + tests. Stop before setup_gpu.sh's prepare_data step, which
#    hardcodes the old range, or edit that line first.
bash setup_gpu.sh

# 3. dataset with the wide range
export TB_LENGTH_REWARD=exact
python prepare_data.py --num-samples 8000 \
    --min-budget 200 --max-budget 3600 --out data/train

# 4. train, detached, so a dropped connection cannot kill it
nohup python train_grpo.py --output-dir checkpoints/lcpo-exact \
    --max-steps 700 --micro-batch 2 --vllm-mem 0.20 > train.log 2>&1 &

# 5. pipeline waits for training to exit, then evals, plots and uploads
tmux new -d -s pipeline 'bash /workspace/pipeline.sh 2>&1 | tee /workspace/pipeline.log'
```

Do not use `rent_and_run.sh` for this. It streams the pipeline in the foreground under a
destroy-on-EXIT trap, so a dropped connection destroys the box mid-training.

## The gate (~3.5 h in, $4 spent)

`save_steps=25`, so checkpoint-250 exists. Stop, measure, decide:

```bash
pkill -f '[t]rain_grpo.py'
python eval_budget.py --lora checkpoints/lcpo-exact/checkpoint-250 \
    --datasets math500 --budgets 256 1024 3600 --limit 50 \
    --out results/gate_250.json
```

- **spread ≥ 1.3x** — the range was the answer. Resume with `--resume`, let it finish.
- **spread ~1.0x** — it was not. Destroy the box. $4 spent instead of $16, and the next
  variable is α ≈ 1e-3, then LoRA rank, then a full fine-tune from DeepScaleR-1.5B-Preview.

Run 3's spread at checkpoint 250 was 1.164x for reference, but that was measured on the
narrow range — not directly comparable, which is why the gate uses 3600 as its top budget.

## Gotchas

- `pipeline.sh` recaptures `results/base.json` only when the *wording* changes, not when the
  budget set changes. The new eval includes 3600; a stale base.json will not have it. A fresh
  box has an empty `results/`, so this resolves itself — just do not rsync an old one up.
- Use the direct `public_ipaddr:direct_port_start`. The `sshN.vast.ai` proxy silently hangs
  on some boxes.
- Five eval budgets instead of four: eval and the checkpoint sweep both run ~25% longer than
  run 3.
- Watch `rewards/length_reward/mean` in W&B, but do not read it as evidence of a dial — it
  improves with uniform compression too. Only a budget-spread eval separates the two.

## If it works

The model card, README and BLOG.md all currently state that the dial does not form and that
this is a compression result. All three need rewriting, and the HF adapter needs replacing —
`upload_hf.py` globs `results/*.json` at the top level while `results/` is now per-run
subdirectories, so pass `--results results/<run-dir>` or fix the glob first.

## If it does not

That is still the third independent confirmation that LCPO-Exact does not transfer to a
rank-32 adapter on the raw distill, and the write-up gets stronger, not weaker: the range was
the paper's own setting and it still was not enough, which points squarely at capacity and
the missing RL stage.
