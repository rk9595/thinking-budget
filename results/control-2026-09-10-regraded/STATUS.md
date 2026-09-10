# Reference comparison: correctness audit

These are locally regraded copies of the 2026-09-09 CUDA generations in
`../control-2026-09-08/`, not a new GPU experiment. Regrading accounts for `<think>` supplied by
the prompt, so unfinished reasoning cannot receive credit for an intermediate boxed answer.
Each sample preserves `legacy_correct`; manifests record original source hashes and the audit.

Only one base-Exact grade changed: high-budget accuracy is 84%, not the original 86%.
The released L1 reference still passes all six predeclared checks. Token lengths, stopping
reasons, and budget response are unchanged. `comparison.json` is the current reference report.
