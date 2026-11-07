"""Freeze a longer seed-zero experiment from published evidence, without model execution.

The source checkpoint hashes are already public. This command constructs data and
timing metadata only; it must not evaluate fresh outcomes or change the parent run.
"""

import json
from pathlib import Path

from novelty_es.data import grade, load_data, prepare_data


def main():
    """Generate the registered fresh endpoint and derive the conservative timing bound."""
    parent = Path("artifacts/pilot")
    experiment = json.loads((parent / "experiment.json").read_text(encoding="utf-8"))
    evidence = json.loads((parent / "artifact-manifest.json").read_text(encoding="utf-8"))
    data = load_data(Path("data"))
    excluded = {tuple(sorted(row["numbers"])) for rows in data.values() for row in rows}
    destination = Path("data/continuation")
    if destination.exists():
        raise FileExistsError("Preserve the frozen fresh data; do not regenerate it implicitly")
    fresh = prepare_data(destination, 20261108, {"fresh": 2048}, excluded)
    fresh["excluded_source_manifest"] = experiment["dataset_manifest"]
    fresh["excluded_multisets"] = len(excluded)
    (destination / "manifest.json").write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    rows = load_data(destination)["fresh"]
    assert not excluded & {tuple(sorted(row["numbers"])) for row in rows}
    assert all(grade(row["witness"], row["numbers"], row["target"]).correct for row in rows)
    metrics = {
        name: json.loads((parent / name / "metrics.json").read_text(encoding="utf-8"))
        for name in ("main", "reward_only")
    }
    checkpoint_seconds = experiment["calibration"]["checkpoint_seconds"]
    maximum = max(row["seconds"] for values in metrics.values() for row in values)
    config = dict(experiment["config"])
    config.update(
        experiment="continuation-seed0-gen100",
        train_seconds=13 * 3600,
        final_seconds=4 * 3600,
        total_seconds=30 * 3600,
        primary_split="fresh",
        publication_suffix="seed0-gen100",
        artifact_repo="novelty-es-continuation-seed0-gen100-artifacts",
        continuation={
            "parent_directory": "artifacts/pilot",
            "parent_generation": 19,
            "target_generation": 100,
            "fresh_directory": destination.as_posix(),
            "fresh_manifest": fresh,
            "parent_files": {
                name: evidence["files"][name]
                for name in (
                    "experiment.json", "main/checkpoint.pt", "reward_only/checkpoint.pt",
                    "original/evaluation.json", "main/metrics.json", "reward_only/metrics.json",
                )
            },
            "pretrained_allocation_deadline": experiment["started_at"] + 5 * 24 * 3600,
            "timing_rule": {
                "observed_max_generation_seconds": maximum,
                "checkpoint_seconds": checkpoint_seconds,
                "estimated_generation_seconds": maximum + checkpoint_seconds,
                "added_generations": 81,
                "predicted_training_seconds_per_arm": 81 * (maximum + checkpoint_seconds),
                "fraction_of_training_cap": 81 * (maximum + checkpoint_seconds) / (13 * 3600),
                "rule": "Fixed target 100; 81 added updates fit 80% of each 13-hour cap using the parent maximum plus checkpoint cost.",
            },
        },
    )
    assert config["continuation"]["timing_rule"]["fraction_of_training_cap"] <= config["timing_fraction"]
    Path("configs/continuation.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fresh": fresh, "timing": config["continuation"]["timing_rule"]}, indent=2))


if __name__ == "__main__":
    main()
