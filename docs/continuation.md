# Running the longer paired continuation

The [follow-up registration](../reports/preregistration-continuation.md) extends the two published seed-zero checkpoints, preserving their FP32 weights, archives, RNGs and global generation indices. It does not restart the original pilot or change its deadline. [Configuration and parent hashes](../configs/continuation.json) fix the source files, target and fresh endpoint before execution.

The OS interruption outlasted that allowance. The separately [registered recovery](../reports/preregistration-recovery.md) preserves the expired attempt and resumes only unfinished work in another directory:

```powershell
uv run novelty-es recover --config configs/continuation-recovery.json --run-dir D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100-recovery
```

Repeat this exact command after another restart; the recovery's stored clock is never reset. `restart-parent/` contains the immutable interrupted source evidence, and `restart-audit.json` verifies numerical state preservation. The final report must disclose the missed earlier deadline. Use this recovery directory for dashboard, report, audit and publication commands below. The historical first-launch commands are retained for provenance, not as a way to reset an expired run.

From the repository root, restore the immutable parent evidence release if `artifacts/pilot` is not already present:

```powershell
uv sync --locked
uv run hf download BurnyCoder/novelty-es-pilot-artifacts --repo-type dataset --revision 949e2dcb385bc2b7088c40444ef879b76596317d --local-dir artifacts/pilot
uv run novelty-es download
```

The [published parent audit](../reports/publication-audit.json) records this revision and the file hashes checked before each fork. Then run:

```powershell
uv sync --locked
uv run pytest
uv run novelty-es run --config configs/continuation.json --run-dir D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100
```

The same command resumes after a process or OS restart. The file lock prevents duplicate writers. The new experiment records its own absolute deadline, capped by the original five-day pretrained allocation; downtime counts. The checked-in configuration retains that dated cap. A later independent replication needs a separately registered configuration and output directory with its own allowance; never reset an existing run's deadline. Each condition is forked only when its phase begins. The fork audit compares every weight, archive entry, RNG state, cursor and counter exactly with its parent before any new candidate runs. Numerical candidate generation, scoring and optimization are shared with the pilot implementation.

For a live read-only dashboard in another terminal:

```powershell
$env:NOVELTY_DASHBOARD_PORT = '8766'
uv run novelty-es dashboard --config configs/continuation.json --run-dir D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100
```

Open `http://127.0.0.1:8766`. Full prompts and outputs stream to the training terminal and the run's `events.jsonl`. Training progress reports cumulative generations; the first added update is generation 20. No fresh held-out outcomes appear until both training phases have ended. Final evaluation includes the new primary, the old secondary test, hard problems, probes and sampling; the two parent checkpoints receive fresh-only evaluation afterward. The untrained baseline reuses its pinned older evaluations and receives new fresh evaluation.

One command regenerates the continuation Markdown report, plots, paper, model cards and separate Pages snapshot:

```powershell
uv run novelty-es report --run-dir D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100
uv run python scripts/verify_results.py --run-dir D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100
uv run novelty-es publish --run-dir D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100
```

Outputs use `continuation-seed0-gen100` in their names. Model repositories use `seed0-gen100` and share the existing research collection. The original report, paper, checkpoints, models and Pages snapshot remain available. Publication is separate from training and must be verified before expanding the study. Problem-bootstrap intervals omit training-seed variability; a longer single-seed trajectory is not a multi-seed result.

After publication, the audited release can be downloaded and its tables, figures, paper and website regenerated without model execution:

```powershell
$releaseRevision = (Get-Content reports/publication-continuation-seed0-gen100.json -Raw | ConvertFrom-Json).revisions.artifacts
uv run hf download BurnyCoder/novelty-es-continuation-seed0-gen100-artifacts --repo-type dataset --revision $releaseRevision --local-dir artifacts/continuation-release
uv run novelty-es report --run-dir artifacts/continuation-release
uv run python scripts/verify_results.py --run-dir artifacts/continuation-release
```

Restore the original pilot release as above for its pinned lineage checks. The newer release includes the interrupted source under `restart-parent/`, both final FP32 checkpoints, inference exports in the linked model repositories, raw evaluations and replay audits. Configuration paths and the dated allowances describe the recorded machine; an independent future training replication requires a separately registered configuration, not an edited deadline in these artifacts.
