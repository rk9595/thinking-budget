# CUDA wiring check: passed

Two MATH-500 problems, three budgets, four arms, seed 42, batch size 8, source `37d7d5c`.
This check established CUDA/vLLM execution, adapter loading, chat-template matching, response
recording, and report generation. Its sample is too small to establish model performance.

The subsequent 50-problem experiment in `../control-2026-09-08/` uses a separately frozen
batch size of 32 and is the relevant reference-control diagnostic. Do not pool these samples
with that experiment: the problems overlap and the batching configuration differs.
