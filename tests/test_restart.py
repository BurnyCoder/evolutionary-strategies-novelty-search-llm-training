"""Recovery is a new registered clock; it never changes its interrupted source."""

import json
import time

import pytest
from test_pipeline import FakeRunner

from novelty_es.continuation import equal_state, file_hash
from novelty_es.data import generate_dataset
from novelty_es.restart import initialize_recovery, recovery_files
from novelty_es.runtime import atomic_json, load_checkpoint
from novelty_es.training import train


def test_recovery_preserves_pending_candidates_and_matches_uninterrupted(tmp_path):
    """Exercise the real train entry point across a new clock and a partial generation."""
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    config.update(population=4, probe_count=2, rotating_count=4, batch_size=2,
                  validation_count=4, validation_every=1)
    data = generate_dataset(5, {"train": 20, "validation": 6, "test": 8, "hard": 4})
    deadline = time.time() + 120
    source, destination = tmp_path / "source", tmp_path / "recovered"
    experiment = {"deadline": deadline, "started_at": deadline - 120, "phase": "reward_only",
                  "dataset_manifest": {}, "code_revision": "original", "config": config,
                  "registered_config": config, "conditions": {}, "failures": [],
                  "phases": {name: {"started_at": deadline - 120, "deadline": deadline}
                             for name in ("main", "reward_only")}, "calibration": {"generations": 2}}
    for name in ("main", "reward_only"):
        runner = FakeRunner(config, source / name, fail_candidate=2 if name == "reward_only" else None)
        if name == "main":
            experiment["conditions"][name] = train(runner, data, source / name, name, 2, deadline, experiment)
            experiment["phases"][name]["finished_at"] = deadline - 1
        else:
            with pytest.raises(RuntimeError, match="simulated interruption"):
                train(runner, data, source / name, name, 2, deadline, experiment)
            atomic_json(source / name / "metrics.json", [])
        runner.log.close()
    atomic_json(source / "experiment.json", experiment)
    (source / "events.jsonl").write_text('{"timestamp":"2026-11-11T21:00:00+00:00"}\n')
    hashes = recovery_files(source)
    amended = {**config, "recovery": {"source_directory": str(source), "files": hashes,
               "week_deadline": deadline + 1000, "last_recorded_event": "2026-11-11T21:00:00+00:00"},
               "total_seconds": 600, "train_seconds": 300, "final_seconds": 300}
    with pytest.raises(ValueError, match="separate"):
        initialize_recovery(amended, source, "recovery-code", now=deadline + 1)
    changed = {**amended, "sigma": 0.2}
    with pytest.raises(ValueError, match="configuration"):
        initialize_recovery(changed, tmp_path / "bad", "recovery-code", now=deadline + 1)
    recovered = initialize_recovery(amended, destination, "recovery-code", now=deadline + 1)
    before = load_checkpoint(source / "reward_only/pending.pt")
    after = load_checkpoint(destination / "reward_only/pending.pt")
    assert equal_state(before["candidates"], after["candidates"])
    assert equal_state(before["rng_state"], after["rng_state"])
    assert len(after["candidates"]) == 2
    assert recovered["deadline"] == deadline + 601
    assert initialize_recovery(amended, destination, "recovery-code", now=deadline + 50) == recovered
    assert recovery_files(source) == hashes
    runner = FakeRunner(amended, destination)
    train(runner, data, destination / "reward_only", "reward_only", 2,
          recovered["phases"]["reward_only"]["deadline"], recovered)
    runner.log.close()
    runner = FakeRunner(config, tmp_path / "continuous")
    train(runner, data, tmp_path / "continuous", "reward_only", 2, deadline, experiment)
    runner.log.close()
    expected = load_checkpoint(tmp_path / "continuous/checkpoint.pt")
    actual = load_checkpoint(destination / "reward_only/checkpoint.pt")
    for key in ("master", "memory", "generation", "data_cursor", "generated_tokens", "responses"):
        assert equal_state(expected[key], actual[key]), key
    assert file_hash(source / "experiment.json") == hashes["experiment.json"]["sha256"]
    assert all(all(item.values()) for item in recovered["recovery"]["preserved_exactly"].values())
