"""Audit the actually interrupted original-model batch using persisted evidence.

This compares decoded text, token IDs and grades, excluding measured duration.
It is evidence for this observed CUDA replay, not for all machines or restarts.
"""

import hashlib
import json
from pathlib import Path

from novelty_es.runtime import atomic_json, utc_now


def main():
    """The failed atomic write contained one uncommitted inference batch."""
    directory = Path("artifacts/pilot")
    before_path = directory / "failure-evaluation-uncommitted.json"
    after_path = directory / "original/evaluation.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))["sets"]["sampled"]
    after = json.loads(after_path.read_text(encoding="utf-8"))["sets"]["sampled"]
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    batch = experiment["calibration"]["batch_size"]
    assert len(after) >= len(before) >= batch
    assert len(before) % batch == 0
    for previous, resumed in zip(before, after, strict=False):
        assert {k: v for k, v in previous.items() if k != "batch_seconds"} == {
            k: v for k, v in resumed.items() if k != "batch_seconds"
        }, "Replayed generation fields differ"
    result = {
        "checked_at": utc_now(), "condition": "original", "split": "sampled",
        "before_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
        "before_rows": len(before), "after_rows_at_check": len(after),
        "replayed_batch_start": len(before) - batch, "replayed_batch_rows": batch,
        "all_prior_and_replayed_generation_fields_equal": True,
        "excluded_field": "batch_seconds", "absolute_deadline": experiment["deadline"],
        "scope": "Observed storage-failure recovery; not a claim about arbitrary restarts or environments",
    }
    atomic_json(directory / "recovery-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
