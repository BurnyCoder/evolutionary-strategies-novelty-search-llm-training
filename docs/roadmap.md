# Implementation and study gates

- [x] Read the closest LLM novelty literature, equations and selected source.
- [x] Pin source/model revisions and implement tested arithmetic/data primitives.
- [x] Publish this registration and dataset manifest before model execution.
- [x] Publish and test the complete resumable pilot pipeline before GPU calibration.
- [x] Calibrate, freeze the common generation count, and evaluate the original model.
- [x] Run main seed 0 and matched reward-only seed 0 within the persisted deadline.
- [x] Evaluate, generate all reports/figures/PDF, inspect outputs, and publish artifacts.
- [x] Preregister the user-requested longer paired continuation and fresh endpoint after reviewing the pilot.
- [x] Preserve the interrupted attempt and publish a tested, separately registered OS recovery.
- [ ] Continue both existing seed-zero trajectories to generation 100, evaluate and publish separately.
- [ ] Review the pilot before preregistering additional pretrained seeds/diagnostics.
- [ ] Freeze and implement the separate byte-level TinyStories study before running it.

The pilot takes precedence. Deferred stages are retained; they are not represented as completed or silently launched while the first result is unreported.

The outage consumed several days of the original week. Recovery takes precedence over starting another study. The remaining independent seeds, diagnostic controls and scratch study have not been executed; their original combined time allocations no longer fit the remaining window. Any reduced scope or new allowance must be registered before those results exist.

The completed pilot's [publication audit](../reports/publication-audit.json) verifies the immutable model/evidence revisions and deployed results. Its [timing deviation](../reports/analysis-notes.md#publication-timing-deviation) records the raw-checkpoint upload overrun separately from the unchanged absolute experiment allowance.
