"""Compare naturally replayed GPU responses across the OS restart; run no models."""

import argparse
import json
from datetime import datetime, timezone

from novelty_es.runtime import atomic_json


def main():
    """Audit unfinished candidate rows present both before interruption and after recovery."""
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    directory = parser.parse_args().run_dir
    state = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    recovery = state["recovery"]
    generation, candidate = recovery["pending_generation"], recovery["saved_candidates"]
    sets = []
    for source in (directory / "restart-parent/events.jsonl", directory / "events.jsonl"):
        rows = {}
        with source.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break  # A live writer may not have flushed its final line yet.
                event = json.loads(line)
                if (event.get("event") == "response" and event.get("phase") == "train"
                        and event.get("condition") == "reward_only" and event.get("generation") == generation
                        and event.get("candidate") == candidate):
                    if source == directory / "events.jsonl" and datetime.fromisoformat(event["timestamp"]).timestamp() < state["started_at"]:
                        continue
                    rows[event["problem_id"]] = {key: value for key, value in event.items()
                                                 if key not in ("timestamp", "batch_seconds")}
        sets.append(rows)
    before, after = sets
    if not before or not before.keys() <= after.keys():
        raise RuntimeError("No complete replay comparison is available yet")
    mismatches = [key for key in before if before[key] != after[key]]
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "generation_index": generation,
              "candidate_index": candidate, "replayed_rows": len(before), "mismatches": mismatches,
              "all_exact": not mismatches, "ignored_fields": ["timestamp", "batch_seconds"],
              "source_trace_sha256": recovery["files"]["events.jsonl"]["sha256"],
              "scope": "Actual GPU response/token/verifier replay for the unfinished candidate; not a full trajectory replay."}
    atomic_json(directory / "restart-replay-audit.json", result)
    print(json.dumps(result, indent=2))
    if mismatches:
        raise RuntimeError("GPU responses differ across the restart; disclose before interpreting results")


if __name__ == "__main__":
    main()
