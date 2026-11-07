# Implementation audit before model execution

2026-11-07. This is a software/protocol audit, not an LLM experiment result.

The source-backed preregistration and exact dataset manifest were published in commit `81fcbb9` before model execution. Unit and end-to-end tests use symbolic arithmetic, direct vector calculations and a deliberately tiny deterministic stand-in; those test outputs are not capability evidence and are not included in pilot result tables.

The bounded code review checked the following failure paths:

- Exact rational AST grading rejects missing/duplicate literals, function calls, exponentiation, Boolean/float constants and division by zero. Correctness is a separate Boolean, not positive reward.
- Every number multiset is unique across all generated splits. Prompts omit solvability witnesses. Probe IDs are excluded from rotating examples.
- Average ranks eliminate arbitrary tie order. Full-weight updates follow the registered effective-alpha convention; every candidate is reconstructed from FP32 masters, and shared tensors are updated once.
- Novelty scoring does not mutate memory. Only correct sibling/history embeddings enter the reference pool, and self-comparisons are excluded. Empty-pool novelty is defined explicitly. Seeded archive insertion and checkpoint recovery are covered by independent expected-value and replay tests.
- Atomic replacement preserves the previous valid checkpoint on simulated disk failure. A population interrupted after partial candidate evaluation resumes to identical final weights/archive on the test environment. Stale pending-generation RNG state is not applied after a committed generation.
- Evaluation persistence stores completed batches and fixed sample IDs; bootstrap pairing aligns problem IDs. The pass@k estimator is checked against exact edge cases. Test data never determine model selection or timing.
- The local HTTP server is read-only, loopback-bound and route-allowlisted. Model generation never executes generated code. HF publishing uploads an explicit set of model/evaluation files. Git ignores `.env`, caches and raw local run files. A process lock prevents two training writers using the same run directory.

The initial report/PDF was generated without empirical results to verify the artifact path. The live dashboard was inspected in a browser and its measure selector was exercised. Pinned downloads finished before the clock. No GPU LLM inference or training was performed during these checks.

Remaining hardware-dependent risks are resolved by the registered calibration: actual CUDA model execution, device memory sufficiency, deterministic noise behavior and measured throughput. If a material implementation error is found after model execution, preserve the failure log and publish an amendment rather than concealing it or resetting the deadline. Full-seed robustness and the deferred scratch study remain outside this first pilot.
