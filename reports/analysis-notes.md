# Analysis notes added during the pilot

## 2026-11-07: descriptive intervention and resource checks

The original-model evaluation was in progress and partial outputs had been inspected when these reporting checks were added. Neither trained arm had started. These checks are therefore labeled post hoc; they do not change the registered primary endpoint, confidence interval procedure, model execution, prompts, grading, timing rule, generation count or optimization.

The intervention check reads the candidate quality and novelty vectors already saved by the published training code. On each observed population, it compares the centered rank weights from quality plus the registered novelty coefficient with the weights from quality alone, using the same float32 fitness arithmetic and tie-aware rank function. It reports positive novelty separately from changed update weights: a common bonus may be positive yet leave every rank unchanged. This is a local counterfactual on recorded candidates, not a simulated alternative training trajectory or an additional significance test.

The resource check streams final response events to count recorded generated tokens by condition and phase. It includes logged retries and abandoned batches. Table columns for committed training candidates and persisted evaluation rows are narrower and explicitly labeled. Partial decoding events are not counted again.

The report also explains a limitation of the preregistered percentile bootstrap: an empirical distribution with no observed successes, or a paired difference with no discordant outcomes, can produce a zero-width interval. Such an interval does not prove a population success probability of zero or general equivalence. The registered estimator remains unchanged.

These additions are implemented in `analysis.py` and `reporting.py`; the public commit history records their timing. Numerical diagnostics are generated from artifacts by the existing report command.
