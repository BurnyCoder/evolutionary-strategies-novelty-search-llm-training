# Restore and inspect the completed studies

These commands use existing evidence only. They perform no training, resumed training or new model evaluation. Both historical allowances have expired. The [continuation registration](../reports/preregistration-continuation.md), [recovery registration](../reports/preregistration-recovery.md) and original configuration/deadlines remain unchanged.

The original attempt stopped during control training. Recovery retained the completed main model and continued the unfinished control under a separately registered allowance. The old attempt remains incomplete under its own deadline. See the [generated report](../reports/experiment-continuation-seed0-gen100.md) and [dated corrections](../reports/independent-corrections-2026-11-12.md) for the evidence and its limits.

## Restore immutable releases

Run from the repository root. [snapshot_download](https://huggingface.co/docs/huggingface_hub/guides/download) restores the evidence datasets **and the separate model repositories**. The BF16 exports are needed to compare tensors with saved FP32 masters; downloading only an evidence dataset is insufficient.

```powershell
uv sync --locked
uv run python scripts/restore_evidence.py --output-root artifacts/restored --cache-dir D:/AI/novelty-hub-cache
```

The script reads pinned revisions from the original [pilot receipt](../reports/publication.json) and [continuation receipt](../reports/publication-continuation-seed0-gen100.json). It does not follow mutable model branches. Use `--dry-run` to inspect the exact downloads. Choose another short writable cache path on other machines. The model downloads preserve their upstream licenses. Corrections are new documentation revisions linked to these predecessors.

This restores the public releases, not every historical local working file. The [restoration check](../reports/independent-restoration-2026-11-12.json) lists local-only calibration, pending-state, console/service-log and upload-receipt files. They are not needed by the read-only verification and reporting commands below; all files shared by the restored and local evidence matched byte for byte.

## Verify without modifying evidence

```powershell
uv run novelty-es verify --pilot-dir artifacts/restored/pilot --continuation-dir artifacts/restored/continuation --data-dir data --fresh-dir data/continuation --output-root output/verification
```

All input and output paths are explicit. The verifier rejects overlapping output/input trees, hashes the inputs before and after, regrades saved responses using an independent exact arithmetic parser, verifies evaluation coverage and dataset separation, reconciles training records and coefficients, recomputes the primary intervals, and compares exported tensors with the saved masters. It constructs no model. Historical audit success flags are not its inputs. Missing generation-zero tensors and missing intermediate embeddings are reported as evidence gaps.

## Reproduce the independent scientific audit

Use the restored directories from the preceding steps. These commands use the project environment and saved records only. The first command checks the historical scoring, tensors and schedules. The second independently decodes tokens, checks rendered prompts and grades responses under a balanced-tag reading. The provenance check uses the GitHub CLI and public Hub metadata.

```powershell
$pilot = 'artifacts/restored/pilot'
$recovered = 'artifacts/restored/continuation'
$checks = 'output/audit-2026-11-26'
uv run novelty-es verify --pilot-dir $pilot --continuation-dir $recovered --data-dir data --fresh-dir data/continuation --output-root "$checks/verification"
uv run python scripts/audit_records.py --pilot-dir $pilot --continuation-dir $recovered --data-dir data --fresh-dir data/continuation --output "$checks/registered-rule-evidence.json"
uv run python scripts/audit_archive.py --pilot-dir $pilot --continuation-dir $recovered --data-dir data --output "$checks/archive.json"
uv run python scripts/audit_provenance.py --pilot-dir $pilot --continuation-dir $recovered --verification "$checks/verification/verification.json" --audit "$checks/registered-rule-evidence.json" --output "$checks/provenance.json"
uv run python scripts/check_embedding_inputs.py --pilot-dir $pilot --continuation-dir $recovered --output "$checks/embedding-inputs.json"
uv run python scripts/audit_upstream.py --output "$checks/source-conformance.json"
uv run novelty-es report-all --pilot-dir $pilot --continuation-dir $recovered --audit-evidence "$checks/registered-rule-evidence.json" --output-root "$checks/release"
uv run python scripts/verify_scientific_release.py --pilot-dir $pilot --continuation-dir $recovered --verification "$checks/verification/verification.json" --release-dir "$checks/release" --output "$checks/presentations.json"
```

For the embedding-length calculation, install the pinned MiniLM tokenizer and sentence configuration in the local Hub cache if needed. Only text assets are required. The dated [review](../reports/reviews/scientific-audit-2026-11-26.md) links the row-level evidence and claim map.

## Regenerate both releases together

```powershell
uv run novelty-es report-all --pilot-dir artifacts/restored/pilot --continuation-dir artifacts/restored/continuation --output-root output/regenerated
```

This stages both reports, their aggregate, figures, papers, model/evidence cards and website snapshots under `output/regenerated`. It does not overwrite the restored runs. Generated cards and summaries are under `releases/`; the two papers are under `output/pdf/` within that output tree. A per-study summary registry keeps the aggregate independent of regeneration order. To regenerate a single study in the same staged tree, use `report --run-dir PATH --output-root output/regenerated`.

## Read-only dashboard

For the completed local recovery already on this machine:

```powershell
$env:NOVELTY_DASHBOARD_PORT = '8766'
uv run novelty-es dashboard --run-dir D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100-recovery
```

For a restored copy, replace `--run-dir` with `artifacts/restored/continuation`. Open `http://127.0.0.1:8766`. The dashboard reads saved progress and response records. Fresh greedy accuracy is the continuation primary endpoint; old-test sampled pass@k is secondary. Original-model old-test, hard, probe and sampled results are reused from the pilot. Parent models have fresh-only evaluations. These are not additional training seeds.

Historical launch and recovery commands remain in the frozen registrations and Git history. They document what ran; they are not restoration instructions. An independent future training replication would need its own preregistration and allowance.
