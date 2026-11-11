"""Pin interrupted artifacts and a recovery allowance before restarting any model."""

import json
from pathlib import Path

from novelty_es.restart import recovery_files
from novelty_es.runtime import load_checkpoint


def main():
    """Record what survived, including the pending population, without inference."""
    source = Path("D:/AI/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100")
    output = Path("configs/continuation-recovery.json")
    if output.exists():
        raise FileExistsError("Preserve the registered recovery configuration")
    previous = json.loads((source / "experiment.json").read_text())
    config = dict(previous["config"])
    generations = {}
    for name in ("main", "reward_only"):
        saved = load_checkpoint(source / name / "checkpoint.pt")
        generations[name] = saved["generation"]
        del saved
    pending = load_checkpoint(source / "reward_only/pending.pt")
    last = None
    with (source / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            last = json.loads(line)["timestamp"]
    config.update(train_seconds=7 * 3600, final_seconds=3 * 3600, total_seconds=10 * 3600,
                  recovery={"source_directory": source.as_posix(), "files": recovery_files(source),
                            "week_deadline": previous["continuation"]["parent_started_at"] + 7 * 86400,
                            "source_deadline": previous["deadline"], "source_generations": generations,
                            "pending_generation": pending["generation"],
                            "saved_candidates": len(pending["candidates"]), "last_recorded_event": last,
                            "reason": "User requested continuation after an OS restart; original allowance expired during the outage."})
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in config["recovery"].items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
