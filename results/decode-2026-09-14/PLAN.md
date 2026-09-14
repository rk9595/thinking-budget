# Decoding-only replication: bounded next step

This follows the [small SFT diagnostic](../sanity-2026-09-13/STATUS.md). Both students
learned a budget response but failed the predeclared length-precision gate. The larger
training stage remains blocked. No new training is part of this experiment.

## Frozen comparison

- Same 24 training problems and 12 separate, filtered-source probes; original 100-problem
  holdout untouched. No new teacher generation and no checkpoint updates.
- Same frozen base and both checksum-verified saved adapters.
- Primary replication: temperature 0.6, fresh seeds 43 and 44, all three arms.
- Secondary diagnostic: greedy decoding (temperature 0), seed 42, all three arms.
- Same exact-budget wording, native template, budgets 512/1024/3600, batch size 36,
  top-p setting 0.95, and a common 8192-token cap. vLLM disables stochastic filtering
  under greedy decoding; it is a different decoding policy, not a replacement seed.
- 972 new responses: 36 problems × 3 budgets × 3 arms × (2 sampled draws + 1 greedy draw).

The machine-readable [plan](plan.json) was committed before GPU provisioning. Every
generation retains raw text, token IDs, finish reason, grading version, model/adapter
hashes, and the sampling configuration. The analyzer validates the complete grids and
matched settings, and reports per-seed training gates separately from probe performance.

## Decision rule

Replicated training-set success requires the **same adapter to pass every existing check
separately at both fresh sampling seeds**. Passing greedy only is evidence of decoding
sensitivity, not replication of the original sampling result. Neither outcome changes the
original failed seed-42 gate or automatically authorizes larger training.

If neither adapter replicates nor passes the greedy training diagnostic, recommend pausing
this recipe rather than another training run. Preserve and document the negative result.
If a signal survives, consider a separately agreed generalization test before scaling:
unseen problems, unseen budget values, answer retention, and comparison with the working
reference. Training-set success alone is insufficient.

## Spending and lifecycle

Additional limit: **$1.50**, inside the remaining original $3 diagnostic allocation.
Prior stage-one charges were $0.519; the two diagnostic limits together do not authorize
spending the conditional $5 larger-training allocation. The new A100 quote is approximately
$0.7211/hour including 60 GB storage, with separate low transfer fees.

The local supervisor begins cleanup at 50 minutes from creation, and a separate
restricted-key server-side timeout requests compute stop at 60 minutes. Provisioning has
a 15-minute local timeout. On normal completion, all artifacts are copied and checksum-verified
before destruction; provider absence must be verified. A stopped instance still incurs
storage costs, and these safeguards are not a provider-enforced hard billing cap.

## Is the project worth continuing?

For a learning/research/portfolio project, one bounded diagnostic is justified by the
new budget-response signal and the low experimental cost. The reproducible comparisons,
failed hypotheses, data validation, and spend-safe execution are useful deliverables even
if this recipe is stopped.

For a deployable model, generalization and reliability are not yet established. A working
[released L1 reference](https://huggingface.co/l3lab/L1-Qwen-1.5B-Exact) already exists and
passed this repository's matched control. The current evidence does not justify claiming
a better model, commercial differentiation, or an unlimited custom-training effort.

Sampling semantics: [vLLM SamplingParams documentation](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/).
