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

## Verify without modifying evidence

```powershell
uv run novelty-es verify --pilot-dir artifacts/restored/pilot --continuation-dir artifacts/restored/continuation --data-dir data --fresh-dir data/continuation --output-root output/verification
```

All input and output paths are explicit. The verifier rejects overlapping output/input trees, hashes the inputs before and after, regrades saved responses using an independent exact arithmetic parser, verifies evaluation coverage and dataset separation, reconciles training records and coefficients, recomputes the primary intervals, and compares exported tensors with the saved masters. It constructs no model. Historical audit success flags are not its inputs. Missing generation-zero tensors and missing intermediate embeddings are reported as evidence gaps.

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
