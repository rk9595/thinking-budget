# Three GRPO runs to give a 1.5B model a thinking-budget dial. I got a 3.2x compression instead.

**TL;DR.** I tried to reproduce L1/LCPO — RL a reasoning model so that "Think for exactly N
tokens" actually controls how long it thinks. Across three GRPO runs and 31 hours of A100
time on DeepSeek-R1-Distill-Qwen-1.5B the dial never formed: output length stayed flat at ~1.08x
spread across budgets from 256 to 2048. What I got instead was a real and useful result —
**MATH-500 reasoning compressed 2524 → 787 tokens (3.2x) for 2.0 accuracy points** — plus a
fairly precise diagnosis of *why* the conditioning failed. This is a writeup of all three
runs, including the two that failed and the bugs that caused them.

Model: [rk9595/thinking-budget-qwen1.5b-lcpo](https://huggingface.co/rk9595/thinking-budget-qwen1.5b-lcpo).

---

## Why this project

I had spent months reading about post-training — RLHF, GRPO, inference-time scaling — without
having actually run any of it. Reading papers gives you vocabulary, not calibration. You
don't learn what `kl = 6e-4` feels like, or that a wrong learning rate looks exactly like a
working run for eight hours, until you burn a GPU rental on it.

So I wanted one project with three properties: a **measurable knob** (not "does it feel
better?"), a **clean before/after curve**, and a shipped artifact on HuggingFace. Test-time
compute control fits all three. If the model works, `N` goes up and tokens go up. If it
doesn't, that's visible in one plot. There is nowhere to hide.

The target was [L1 (Aggarwal & Welleck, CMU 2025)](https://arxiv.org/abs/2503.04697), which
introduces **LCPO** — Length Controlled Policy Optimization. You put a budget in the prompt,
and you add a length term to the RL reward.

## The recipe

**Base model:** DeepSeek-R1-Distill-Qwen-1.5B. Small enough to train on a single rented
A100, already a reasoning model, and it writes ~2500 tokens per MATH-500 problem unprompted
— plenty of fat to cut.

**Data:** 8,000 problems from DeepScaleR-Preview. Each gets a budget sampled uniformly from
200–2000 tokens, injected into the prompt:

```
{problem}

Let's think step by step and output the final answer within \boxed{}. Think for exactly 731 tokens.
```

**Reward.** Two terms, summed:

```python
r = 1[answer correct] - α · |N - tokens_used|      # LCPO-Exact
r = 1[answer correct] - α · max(0, tokens_used - N) # LCPO-Max
```

with α = 3e-4 and correctness checked by `math-verify` against the boxed answer. The
distinction between the two variants turns out to be the whole story, so hold onto it:

- **Exact** treats the budget as a *target*. Writing 300 tokens when asked for 1000 is
  penalized just as much as writing 1700. This is what should force the model to *read* N.
- **Max** treats the budget as a *ceiling*. Finishing early is free.

**Algorithm:** GRPO via TRL, 8 generations per prompt, effective batch 32, LoRA r=32 on all
attention and MLP projections, vLLM in colocate mode for rollouts. One A100 80GB.

**Eval:** 100 held-out problems each from MATH-500 and GSM8K, at budgets
{256, 512, 1024, 2048}, temperature 0.6, against the unmodified base model. The metric that
matters is **spread**: max mean-tokens over min mean-tokens across budgets. A working dial
gives you a big number. 1.0 means the model is ignoring you.

---

## Run 1 — three bugs in a trenchcoat

1000 steps, 15.1 hours, and completely flat. Mean tokens on MATH-500: 2190 / 2197 / 2225 / 2229
for budgets 512 / 1024 / 2048 / 3600. Accuracy actually went *up* slightly (0.78 → 0.81),
which is the tell — the model had learned essentially nothing about length and drifted a
little on style.

Three independent bugs, any one of which would have been fatal:

**1. The policy never moved.** `grad_norm` sat at ~0.005 and `kl` at ~6.7e-4 for the entire
run; policy entropy drifted only 0.96 → 0.70, versus 0.75 → 0.35 in the run that actually
learned something. I had set `lr = 5e-6` — a sensible full-finetune learning rate that is roughly 10x too
small for a LoRA adapter, where only a low-rank slice of the weights is trainable. The run
completed, logged beautifully, uploaded cleanly, and did nothing. **Fixed: 2e-5.**

**2. The prompt and the reward disagreed.** The prompt said "Think for **maximum** N tokens"
while the reward was LCPO-**Exact**, `-α·|N - n|`. So the model was told the budget was a
ceiling and paid to hit it precisely — including being paid to *pad up* to N. I had
`prepare_data.py`, `eval_budget.py` and `profile_inference.py` each building prompts in
separate processes, and they drifted.

**Fixed structurally, not by hand:** the wording now lives in exactly one place
(`rewards.budget_instruction()`, keyed off a single env var), it gets baked into every
training example, and `train_grpo.py` reads it back off the dataset and refuses to start on
a mismatch. Consistency you enforce at startup beats consistency you remember.

**3. A third of the data pushed the wrong way.** I sampled budgets 100–3600 against a
natural length of ~2500. Every example above 2500 was rewarding the model for writing
*longer*. **Fixed: 200–2000**, entirely below the natural length, so every example points the
same direction, and the 4096 completion cap stays as real headroom.

*Read that last clause again — "every example points the same direction." I wrote it as a
virtue. It is the reason the dial never formed. See [the better suspect](#the-better-suspect-i-made-not-conditioning-cheap-and-l1-didnt)."*

## Run 2 — LCPO-Max, and the degenerate solution

700 steps, 7.6 h. All three fixes landed, and the training metrics were night and day:
`kl` 6e-4 → **0.03**, mean completion length 4096 → **670**, `clipped_ratio` 0.97 → **0**.

Ceiling compliance was real and dramatic. Over-budget rate on MATH-500:

| budget | base | trained |
|---|---|---|
| 512 | 95% | 42% |
| 1024 | 87% | **3%** |
| 2048 | 53% | **0%** |

And it is a completely useless model.

Mean tokens across budgets 256 / 512 / 1024 / 2048: **490 / 559 / 546 / 528.** The model
learned "always be short." It never overshoots 2048 because it never writes more than ~550
tokens about anything.

This is the **degenerate solution of a max-only reward**, and in hindsight it's obvious.
Under `-α·max(0, n - N)`, being shorter than the budget is never penalized. Nothing in the
objective pulls length *up*. The globally optimal policy for the length term alone is "emit
as few tokens as possible", and the only thing holding it back is the correctness term. So
it collapsed — completely, by step 100 — and paid **8 to 13 accuracy points** on MATH-500 for
the privilege.

**Ceiling compliance and a compute dial are different properties, and LCPO-Max can only give
you the first.**

The other lesson from run 2: **training-time correctness did not reveal any of this.** It
went 0.298 → 0.277, which at 32 samples per step is indistinguishable from noise. The
collapse was only visible in held-out eval across budgets. If you are training a controllable
property, your training metrics must include a measure of *control*, not just a measure of
*quality*.

## Run 3 — LCPO-Exact, the real attempt

600 steps, 8.4 h — the last run the rental budget stretched to.

Exact penalizes undershoot too, so the degenerate "always be short" policy is no longer free.
On paper this is the variant that has to produce conditioning. The training curve looked
healthy the whole way: `frac_reward_zero_std` = 0 (GRPO always had real advantages to work
with), `clipped_ratio` = 0 (nothing truncated), mean completion length falling steadily from
4096.

**The dial still did not form. Third time.**

MATH-500 mean tokens across budgets 256 / 512 / 1024 / 2048:

**781 / 783 / 763 / 821** — a **1.08x spread.** GSM8K: 1.06x.

For reference, the *untrained base model's* spread is 1.04x. Statistically, the trained model
ignores the budget number about as thoroughly as the model that was never trained on budgets
at all.

### But it is a genuinely good compression result

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

Averaged: **MATH-500 2524 → 787 tokens (3.2x) for −2.0 accuracy points**; GSM8K 1239 → 457
(2.7x) for −5.2 points.

**Read the tokens column down, not across.** Down the column, it's a 3x efficiency win.
Across budgets, it's constant — and that constancy is the negative result.

Run 2 paid 8–13 points for a *worse* compression. So the two variants separate cleanly:
**Exact ≈ "compress hard, keep accuracy"; Max ≈ "collapse and lose accuracy."** That
comparison is the most useful thing the project produced.

The adherence numbers show the lopsidedness nicely. Mean `|tokens − N|` on MATH-500:

| budget | base | trained |
|---|---|---|
| 256 | 2310 | **525** |
| 512 | 2030 | **341** |
| 1024 | 1531 | **398** |
| 2048 | 1149 | **1278** ← *worse than base* |

A model that always writes ~790 tokens undershoots a 2048 budget almost as badly as the base
model overshoots it.

---

## Why the conditioning never formed

This is the part I actually care about, and I have four pieces of evidence.

### 1. The model regressed to the conditional mean

The per-checkpoint sweep on MATH-500 shows length falling hard while spread does nothing:

| step | 256 | 512 | 1024 | 2048 | spread |
|---|---|---|---|---|---|
| 100 | 2178 | 2176 | 2257 | 2319 | 1.065x |
| 250 | 965 | 861 | 935 | 1002 | 1.164x |
| 500 | 777 | 751 | 828 | 791 | 1.103x |
| 600 | 751 | 731 | 855 | 767 | 1.170x |

Mean length drops 3x. Spread goes nowhere. The model found the strategy that minimizes
expected `|N − n|` **without reading N**: emit the conditional mean of the budget
distribution and eat the variance. Given budgets ~ U(200, 2000), the optimal constant is
around 1100 for the L1 objective and lower once correctness pressure is in the mix. It landed
at ~790. It solved the problem I posed rather than the problem I meant.

### 2. The run was truncated, not converged

I originally read the final `kl` of 0.006 — five times *lower* than run 2's 0.03 — as "Exact's
gradients cancel and the policy is stuck." Pulling the actual curves out of W&B, that's not
right. Binned over 50 steps:

| steps | kl | mean length | length reward | entropy |
|---|---|---|---|---|
| 1–50 | 0.00031 | 3638 | −0.747 | 0.753 |
| 101–150 | 0.00075 | 3294 | −0.679 | 0.671 |
| 201–250 | 0.00409 | 2297 | −0.461 | 0.511 |
| 301–350 | 0.00717 | 1721 | −0.332 | 0.435 |
| 401–450 | 0.00858 | 1628 | −0.311 | 0.366 |
| 501–550 | 0.01058 | 1322 | −0.237 | 0.338 |
| 551–600 | 0.01058 | 1375 | −0.242 | 0.348 |

`kl` is still **climbing** at the end. Mean length is still **falling**. The length reward is
improving monotonically. Run 2 by contrast plateaued at `kl` ≈ 0.032 and ~700 tokens from
step 300 onward and then sat there for 400 more steps.

So Exact isn't stalled. It's moving in the same direction as Max, roughly **3x slower per
step** — which is what partial gradient cancellation actually looks like. High-budget examples
say "write more", low-budget examples say "write less", and those two only reconcile into a
coherent update if the policy attends to N. It doesn't, so they partially cancel and progress
is slow rather than absent.

The `kl = 0.006` I'd been quoting was a single noisy final step, not the trend. Worth
checking your curves before you build a story on your last logged value.

### 3. Compression and conditioning are separate axes, and only one was moving

This is the important consequence. Over 600 steps, **compression improved steadily and
conditioning never started.** Those are independent. Training longer would have bought more
compression — length was still dropping — and there is no evidence it would have bought a
dial, because the conditioning signal was flat from step 100 to step 600.

### 4. Entropy collapsed

Policy entropy fell 0.75 → 0.35 over the run (run 2: 0.72 → 0.27). The model is not
developing a richer, length-aware policy; it is narrowing onto a single length **habit**. A
length *function* of N would need to preserve variability across contexts. This looks like the
opposite.

### The first suspect: α is too small

Final `rewards/length_reward/mean` = **−0.199**, i.e. mean `|n − N|` ≈ 660 tokens. Meanwhile
`rewards/correctness_reward/std` = **0.44**.

Inside a GRPO group, advantages come from *variance*. The length term contributes a
systematic offset of ~0.2 while correctness contributes ~0.44 of spread — so **length is a
minority shareholder in the gradient**. At α = 3e-4, a 660-token miss costs 0.198, which is
less than the value of flipping one answer from wrong to right. The model's rational move is
to optimize correctness and take the cheap constant-length approximation on the side. Which is
exactly what it did.

### The better suspect: I made not-conditioning cheap, and L1 didn't

I went back to the paper to check what I had actually changed. Four things, and I had only
been counting two of them:

| | L1 | my run 3 |
|---|---|---|
| starting model | **DeepScaleR-1.5B-Preview** — already RL fine-tuned on this dataset | DeepSeek-R1-Distill-Qwen-1.5B — the raw distill |
| method | **full fine-tune** | LoRA r=32 |
| budget range | **U(100, 4000)** | U(200, 2000) |
| data | 40K | 8K |
| steps | 700 | 600 |

The budget range is the one I had filed as a *fix*. L1 samples budgets that straddle the
model's natural length; I narrowed mine to sit entirely below it. That decision has a price,
and it is computable.

A model that ignores N and emits a constant pays the mean absolute deviation of the budget
distribution. For `U(a,b)` the best constant is the median and the cost is `(b−a)/4`:

- my range: 1800/4 = **450 tokens**
- L1's range: 3900/4 = **975 tokens**

**Refusing to condition was 2.2x cheaper in my setup than in theirs.** At α = 3e-4 that is a
0.135 penalty against a correctness spread of 0.44 — a minority shareholder, exactly as
above. Under L1's range the same constant strategy costs 0.293, which is the same order as
correctness rather than dominated by it.

And there is a qualitative half that the arithmetic misses. With every budget below the
natural length, **"just be shorter" satisfies every training example simultaneously.** All the
gradients point one way, so a uniform habit is not an approximation — it is a complete
solution. Straddling is what makes the budget informative in both directions: low budgets say
shorten, high budgets say lengthen, and the only policy that satisfies both is one that reads
N. I removed the pressure that makes conditioning necessary, and then spent 8.4 hours
measuring its absence.

This also reframes suspect 1. The penalty for ignoring N is `α · (b−a)/4`, so widening the
range and raising α are the same knob. Widening is free, and it is what the paper did.

**If I ran a fourth time**, the first experiment is the budget range: **U(100, 4000)**, one
flag, no other changes. Then α ≈ 1e-3 if that is not enough. Then capacity — higher LoRA
rank or a full fine-tune, since L1's numbers come from a full fine-tune and a rank-32 adapter
may simply lack the room to represent "read this integer and modulate my stopping behavior
accordingly." Starting from DeepScaleR rather than the raw distill would also remove an
entire RL stage that my run was trying to do simultaneously with length control. The kill
criterion is cheap either way: if budget spread is still under 1.3x at checkpoint 250, stop
there.

I want to be honest about the status of this. It is a hypothesis with arithmetic behind it,
not a finding. The experiment that would settle it costs about nine GPU-hours and I have not
run it.

---

## Engineering notes from three GPU rentals

The ML content above is maybe half of what I actually learned. The rest is operational, and
nobody writes it down.

**On GRPO and TRL specifically:**

- GRPO OOMs at micro-batch 8 on an 80GB A100 for a 1.5B model, which is counterintuitive
  until you remember the logits tensor is `batch × seq × 152k vocab`. Micro-batch 2 with
  gradient accumulation 16 holds the effective batch at 32 and fits in 46GB.
- Do **not** lower `max_completion_length` to save memory. Mean terminated length starts at
  ~2300; a cap below ~3000 truncates nearly everything, which destroys the *in-group reward
  variance* GRPO depends on. You get a run with `frac_reward_zero_std` climbing and no
  learning signal at all.
- TRL names the logged reward metric after your reward function's `__name__`. So the key is
  `rewards/length_reward_max/mean` for one variant and `rewards/length_reward/mean` for the
  other, and a monitoring grep written for one **silently matches nothing** on the other.
- Watch `kl` from the first hundred steps. Still ~1e-4 after a few hundred? The run is
  already dead — kill it. That single check would have saved 15 hours of A100 time.
- TRL 1.x dropped `max_prompt_length` and `warmup_ratio`. Its generation path also NaNs on
  Apple MPS, so local smoke tests have to be pinned to CPU.

**On renting GPUs:**

- vLLM 0.27 on Python 3.11 needs **both** `pip uninstall flashinfer-python` **and**
  `VLLM_USE_FLASHINFER_SAMPLER=0`. Removing the package alone just trades an import crash for
  a `ModuleNotFoundError`, because vLLM's capability check imports the backend unguarded.
- My first orchestration script streamed the whole pipeline in the foreground with a
  destroy-the-instance-on-EXIT trap. A dropped SSH connection would have killed the box
  mid-training. Never put a `trap ... EXIT` teardown on a foreground long-running remote job.
  rsync, `nohup`, tmux, poll.
- vast.ai's proxy host `sshN.vast.ai` silently hangs on some instances. Use the direct
  `public_ipaddr:direct_port_start`, and add `-o BatchMode=yes` so a missing key fails fast
  instead of blocking forever on a password prompt.
- Credentials live in a file outside the repo (mode 600) and go to the box via **stdin**,
  never on an ssh command line where they'd appear in the remote process list.
- Python block-buffers stdout when redirected to a file, so `train.log` shows the tqdm bar
  (stderr, unbuffered) long before the first metric dict. That lag is not a hang. I nearly
  killed a healthy run over it.
- `pgrep -f foo.sh` matches the shell running it. Use `pgrep -f '[f]oo.sh'`.

**On not lying to yourself:**

- Snapshot the code *with* the results. Every run directory here carries a `code/` copy,
  because three runs in, "which version of `rewards.py` produced this?" becomes unanswerable
  otherwise.
- Keep a stale model card and you will ship a lie. Mine described run 2's recipe while
  serving run 3's weights for two days, because the upload script overwrote `results/` but
  not `README.md`. Automate the artifact that humans read, or it will drift.

## What it took

| run | variant | steps | wall clock | outcome |
|---|---|---|---|---|
| 1 | Exact | 1000 | 15.1 h | No effect — three bugs |
| 2 | Max | 700 | 7.6 h | Collapse — "always be short" |
| 3 | Exact | 600 | 8.4 h | 3.2x compression, no dial |

**31 hours** of rented A100 time in total. Run 1 — the one that taught me the most — also
burned close to half of it, entirely because I let a dead run go the full distance instead
of checking `kl` at step 100.

## Was it worth it?

I did not build the thing I set out to build. There is no budget dial.

But "reproduce a paper" was the stated goal, not the actual one. The actual one was to stop
having read about post-training and start having done it. Measured that way: I can now look at
a GRPO run's `kl`, `grad_norm`, `entropy`, `clipped_ratio` and `frac_reward_zero_std` and tell
you within a hundred steps whether it's alive, converging, or collapsing. I know what a
degenerate reward looks like from the inside, and that the degeneracy is obvious in hindsight
and invisible in advance. I know that a reward with cancelling gradients doesn't fail loudly —
it just gets slow, which reads exactly like a healthy run if you're only watching the loss.

And the artifact is real: a 1.5B reasoning model that thinks **3.2x more concisely for 2 points
of accuracy**, with the negative result documented on the model card rather than buried. If you
want shorter reasoning traces from a small model, it works. If you want to control the length,
it doesn't, and the card says so in the first paragraph.

A negative result you can explain is worth considerably more than a positive result you can't.

---

*Model and full artifacts:
[rk9595/thinking-budget-qwen1.5b-lcpo](https://huggingface.co/rk9595/thinking-budget-qwen1.5b-lcpo).
All three runs' evals, per-checkpoint sweeps and training code are in the repo.*
