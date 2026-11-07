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

## Weighting amendment before either trained condition

The user requested source-faithful weighting unless a deviation had a convincing performance rationale. The source audit found no task-specific evidence favoring combined-reward ranks and found an explicit motivation for post-normalization addition in ExTra. [Amendment A](preregistration.md#amendment-a-source-aligned-weighting-before-either-trained-condition) therefore supersedes the old rank diagnostic above: compare `Z(Q) + gamma*N` against `Z(Q)` on each observed population, in the same precision as the optimizer. A shared positive novelty bonus now changes update coefficients; only a bonus lost to floating-point rounding fails to do so. Neither rank-only nor old-coefficient training results exist. Original-model results were already partly observed, so the amendment must not be described as predating all model execution.

Model execution was deliberately interrupted during original-model evaluation to publish and test the amended method before training. Both trained arms start from original pretrained weights, not the discarded calibration model or any amended-in-place training checkpoint. The phase and absolute clocks remain unchanged. Any baseline work that no longer fits the calibration phase is deferred using the existing final-evaluation rule.

## Observed GPU evaluation recovery

The failed temporary evaluation file was preserved before resuming. `uv run python scripts/verify_recovery.py` compares it with the committed original-model evaluation, excluding only elapsed batch time. The observed replay preserved the full decoded responses, token IDs, problem/sample identities and verifier fields exactly. Counts and the failed-file hash are generated in `artifacts/pilot/recovery-audit.json`; the source snapshot is `failure-evaluation-uncommitted.json`. Both are included in artifact publication. This is an observed CUDA evaluation replay after the storage failure, distinct from the stand-in model's interrupted-training tests. It does not establish equivalence across arbitrary hardware, package changes or OS restarts.

## Release reporting during main-arm training

The final report now computes paired differences against the original checkpoint in addition to the registered main-versus-control comparison. These implement the planned baseline comparisons using the same problem-bootstrap estimator and aligned problem IDs. They remain descriptive secondary estimates with unadjusted intervals; they do not replace the primary endpoint. This presentation was added after seeing the original-model evaluation and interim main-arm training/validation, before either trained arm's test evaluation.

Static plots retain a zero baseline and derive a common upper limit for both conditions from all observed values in that panel, including accuracy interval endpoints. Accuracy is labeled as a percentage; generation ticks are integers. Panels use different metric scales, stated explicitly, so small observed values remain readable without selectively omitting lower values. CPU reporting, testing, auditing and backup work overlapped training; reported durations are descriptive rather than isolated system-speed measurements. These changes do not alter any model computation or recorded raw outcome.

Before held-out evaluation of the trained arms, the control's reporting call was corrected to use its actual zero novelty coefficient. The earlier diagnostic applied the main coefficient hypothetically to every recorded population, which could misleadingly describe potential control weight changes as applied changes. The control still reports its computed positive novelty, but correctly reports no novelty-induced update-weight changes. Main-arm diagnostics, raw coefficients and both training trajectories are unchanged. A regression test uses positive control novelty to distinguish this case.
