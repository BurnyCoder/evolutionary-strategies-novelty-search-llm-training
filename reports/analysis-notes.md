# Analysis notes added during the pilot

## 2026-11-07: descriptive intervention and resource checks

The original-model evaluation was in progress and partial outputs had been inspected when these reporting checks were added. Neither trained arm had started. These checks are therefore labeled post hoc; they do not change the registered primary endpoint, confidence interval procedure, model execution, prompts, grading, timing rule, generation count or optimization.

The intervention check reads the candidate quality and novelty vectors already saved by the published training code. On each observed population, it compares the centered rank weights from quality plus the registered novelty coefficient with the weights from quality alone, using the same float32 fitness arithmetic and tie-aware rank function. It reports positive novelty separately from changed update weights: a common bonus may be positive yet leave every rank unchanged. This is a local counterfactual on recorded candidates, not a simulated alternative training trajectory or an additional significance test.

The resource check streams final response events to count recorded generated tokens by condition and phase. It includes logged retries and abandoned batches. Table columns for committed training candidates and persisted evaluation rows are narrower and explicitly labeled. Partial decoding events are not counted again.

The report also explains a limitation of the preregistered percentile bootstrap: an empirical distribution with no observed successes, or a paired difference with no discordant outcomes, can produce a zero-width interval. Such an interval does not prove a population success probability of zero or general equivalence. The registered estimator remains unchanged.

These additions are implemented in `analysis.py` and `reporting.py`; the public commit history records their timing. Numerical diagnostics are generated from artifacts by the existing report command.

## 2026-11-07: Windows atomic-replacement recovery

During original-model sampled evaluation, Windows denied replacement of the evaluation JSON with `WinError 5`. The exception and traceback are retained in the run's event log and experiment state. The previous complete JSON remained readable; neither trained condition had begun. A concurrent reader can prevent a Windows rename when its sharing mode omits deletion sharing; the specific holder was not identified. See the [Windows file-sharing specification](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew).

The storage helper now retries transient Windows access/sharing errors for a bounded interval before surfacing persistent failure. It retains atomic replacement and the previous valid record. Regression tests cover temporary and persistent locks, and a native Windows read-handle experiment verifies recovery after the handle closes. This is a persistence repair, not an optimizer or evaluation-method amendment. Resume uses the last committed batch; the uncommitted batch may be regenerated with its original seed. Downtime, repeated work and retry time consume the original allowance. The failed temporary record is preserved separately for audit.
