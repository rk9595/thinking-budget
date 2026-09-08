# Reference-control experiment: prepared, not yet executed

The 50 MATH-500 development problems and three model revisions are frozen in this directory.
There are four arms because the L1 reference uses different wording and a different chat template;
each trained model has a matching base-model control. No accuracy or budget-adherence results have
been generated for these arms yet.

Local verification on 2026-09-08:

- Reward and evaluation unit checks pass, including rejecting reversed/constant budget responses,
  detecting incomplete sample grids, preserving full SFT reasoning, and rejecting truncated teacher examples.
- A real tiny random model exercised the TRL training loop on CPU: pause at step 1, reject a changed
  learning rate, resume optimizer/scheduler state, and finish step 2. This checks implementation wiring,
  not learned budget control.
- The local dependency lock installs and passes `uv pip check`.
- The Linux CUDA dependency lock resolves but has not yet been installed or run on a CUDA machine.
- Pilot splits were prepared locally: 500 training and 100 separate development problems, with no
  overlapping IDs. A separate GRPO dataset contains 1,500 rows, 500 at each of 512/1024/3600 tokens.

Next action: run the separate two-problem CUDA wiring check, then this 50-problem comparison on
an authorized GPU. Do not proceed to SFT/GRPO based on the local tests alone.
