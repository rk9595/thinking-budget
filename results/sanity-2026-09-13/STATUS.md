# Small SFT learnability diagnostic: completed, scaling gate not passed

The two students now respond strongly to the requested budget on this small diagnostic.
Neither passes the predeclared length-precision requirement. The larger $5 SFT pilot remains
blocked; no GRPO or release was started. This is progress over the earlier failed pilot,
not proof of generalization or a completed project.

## Protocol and evidence

The [frozen plan](plan.json) selected 24 training problems with one verified teacher response
at each of 512/1024/3600 tokens, plus 12 separate diagnostic probes. All came from the verified
teacher-source pool, so the probes are not representative of an unrestricted test set.
The original 100-problem held-out development set was not used.

Both students start from the same frozen DeepSeek-R1-Distill-Qwen-1.5B revision and seed 42.
Each uses rank-32 all-linear LoRA, learning rate 1e-4, microbatch 1, accumulation 16, and
20 epochs / 100 optimizer steps over 72 examples. The only intended arm difference is
token-weighted versus equal-completion NLL. Both use the corrected prompt boundary and
explicit standard NLL path, and both are seeded before LoRA initialization.

Evaluation uses identical prompts, batch size 36, seed 42, temperature 0.6, top-p 0.95,
and an 8192-token cap for every requested budget. All 324 responses are retained:
36 problems × 3 budgets × 3 arms. There is no budget-dependent forced stop.
The Linux GPU dependency check and all 40 then-current tests passed before generation.

## Training-problem results

| Arm | Mean tokens at 512 | At 1024 | At 3600 | Strict gate |
|---|---:|---:|---:|---|
| Base | 3955 | 3992 | 3178 | Control only |
| Token-weighted SFT | 564 | 1523 | 3691 | Fail |
| Equal-completion SFT | 493 | 1200 | 3415 | Fail |

| Arm | Accuracy at 512 / 1024 / 3600 | Mean absolute relative error at 512 / 1024 / 3600 |
|---|---|---|
| Base | 83.3% / 87.5% / 83.3% | 672.5% / 289.9% / 51.6% |
| Token-weighted SFT | 100% / 95.8% / 91.7% | 26.5% / 59.3% / 24.8% |
| Equal-completion SFT | 100% / 100% / 87.5% | 11.3% / 37.5% / 36.3% |

Both students pass six of seven checks. The failing check is **mean absolute per-response
relative length error ≤35% at every budget**. A mean length near the target does not pass
this check when undershoots and overshoots cancel. Thresholds and original outputs are unchanged.

All 24 training problems have longer outputs at 3600 than at 512 for both students, versus
7/24 for the base. All three budgets are nondecreasing on 24/24 problems for equal-completion
SFT and 22/24 for token-weighted SFT. Endpoint slopes are 0.946 and 1.013, respectively,
versus −0.252 for the base. Each student still clips on 2/24 long-budget training answers.

On the 12 separate probes, token-weighted SFT averages 677/1130/2758 tokens with
83.3%/91.7%/91.7% accuracy; equal-completion SFT averages 662/1544/3707 with
83.3%/100%/75% accuracy. Equal-completion weighting is **not uniformly better**:
its long-budget probe output clips on 2/12 problems, versus none for token weighting.
These small, filtered probes do not establish held-out performance.

Both runs finish in about 513 seconds. Their last logged training-batch losses are about
0.011; these teacher-forced losses do not establish stable free-running generation.
The experiment also changes training duration and prompt-boundary handling relative to
the old pilot, so it cannot isolate one cause of improvement over that pilot.

## Remaining errors and next step

The [post-hoc error audit](error_audit.json) preserves all outliers in the reported metrics.
For example, token weighting generates 7630 tokens for a 1024-token request. Equal-completion
weighting generates 3802 and 3191 tokens for two 1024-token requests. Long-budget failures
include repeated checking or repeated answers without completing the reasoning section.
These are observed behaviors, not proof of a specific underlying training defect.

Recommended next experiment: a **small decoding/stopping diagnostic using the saved adapters**,
before more training or data generation. Freeze fresh seeds and compare both adapters and
the matched base at the existing temperature; add greedy decoding only as a separately
labeled diagnostic. Retain the same common 8192-token cap, inspect termination and length
tails, and keep the original 100-problem holdout untouched. This distinguishes sampling
instability from persistent conditioning errors without retraining. Greedy results must not
be substituted for the failed temperature-0.6 gate or used to retroactively change this result.
The follow-up has not been launched; no larger-run authorization is inferred from this near miss.

## Cost, backups, and cleanup

Instance **50898112** (A100-SXM4 40GB) is verified absent from the provider. The independent
supervisor retrieved and SHA-256-verified the results and both final adapters before deletion,
then recorded provider absence at **2026-09-13 15:20:39 UTC**. A fresh check on September 14
again returned no instances. The remote compute-stop timeout was armed as a fallback;
normal verified destruction completed before it was needed.

Provider-reported charges retrieved September 14:

| Item | USD |
|---|---:|
| GPU | 0.496 |
| Storage | 0.012 |
| Download | 0.010 |
| Upload | 0.001 |
| Total | **0.519** |

This is below the $3 first-stage limit. The conditional $5 larger run was not started.
The reported amount is the provider's itemized charge total, not a credit-balance difference.

Local ignored adapters are under `checkpoints/sanity-2026-09-13/{token,sequence}/final/`.
The token adapter SHA-256 is
`c6dec52eccff333df9d081166f4737d268924acc227049aae9c4ab20d076dead`;
the sequence adapter SHA-256 is
`595785b6597d8a488872773145eab139e1498467cab0767f8613c4edb3f1fe4d`.
The [retrieval manifest](artifact_hashes.json) covers both adapters and all completed evaluation
artifacts. The later audit and this report are additional local analysis, not part of that
original retrieval manifest. No adapter or training dataset was published.

Primary result: [comparison.json](comparison.json). Raw outputs, manifests, and summaries
for `base`, `token`, and `sequence` are alongside it. Exit code 2 in
[job_status.json](job_status.json) records a completed experiment with a failed gate, not a crash.
