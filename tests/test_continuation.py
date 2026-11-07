"""Continuation checks use synthetic CPU fixtures, never reported model results."""

import json
import time
from pathlib import Path

import pytest
import torch
from test_pipeline import FakeRunner

from novelty_es.continuation import equal_state, experiment_data, file_hash, fork_checkpoint, validate_parent
from novelty_es.data import generate_dataset, grade, load_data, prepare_data
from novelty_es.evaluation import evaluate
from novelty_es.runtime import atomic_json, load_checkpoint
from novelty_es.training import train


def test_fresh_data_excludes_every_parent_multiset(tmp_path):
    """A fresh endpoint excludes all earlier splits, including permutations and targets."""
    parent = generate_dataset(11, {"train": 40, "test": 20, "hard": 10})
    excluded = {tuple(sorted(row["numbers"])) for rows in parent.values() for row in rows}
    manifest = prepare_data(tmp_path, 11, {"fresh": 24}, excluded)
    fresh = load_data(tmp_path)["fresh"]
    assert manifest["splits"]["fresh"]["count"] == 24
    assert not excluded & {tuple(sorted(row["numbers"])) for row in fresh}
    assert sum(len(row["numbers"]) == 3 for row in fresh) == 12
    assert all(grade(row["witness"], row["numbers"], row["target"]).correct for row in fresh)
    assert generate_dataset(11, {"fresh": 24}, excluded)["fresh"] == fresh


def test_registered_fresh_dataset_and_original_hashes():
    """The checked-in fresh endpoint is disjoint and the original manifest remains valid."""
    config = json.loads(open("configs/continuation.json", encoding="utf-8").read())
    data, manifest = experiment_data(config, Path("data"))
    assert len(data["fresh"]) == manifest["splits"]["fresh"]["count"] == 2048
    assert sum(len(row["numbers"]) == 3 for row in data["fresh"]) == 1024


def test_fork_then_interruption_matches_continuous_training(tmp_path):
    """A real train entry point preserves global schedules, masters, archive and counters."""
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    config.update(population=4, probe_count=2, rotating_count=4, validation_count=4,
                  validation_every=1, batch_size=2, sampled_problem_count=4)
    data = generate_dataset(5, {"train": 20, "validation": 6, "test": 8, "hard": 4, "fresh": 6})
    deadline = time.time() + 120
    experiment = {"deadline": deadline, "dataset_manifest": {"synthetic": True}, "code_revision": "test-parent"}
    continuous = tmp_path / "continuous"
    runner = FakeRunner(config, continuous)
    train(runner, data, continuous, "main", 3, deadline, experiment)
    runner.log.close()
    parent = tmp_path / "parent"
    runner = FakeRunner(config, parent)
    train(runner, data, parent / "main", "main", 1, deadline, experiment)
    runner.log.close()
    source = parent / "main/checkpoint.pt"
    parent_hash = file_hash(source)
    before = load_checkpoint(source)
    extended = {**config, "total_seconds": 240, "continuation": {
        "parent_directory": str(parent), "parent_generation": 1, "target_generation": 3,
        "parent_files": {"main/checkpoint.pt": {"sha256": parent_hash, "bytes": source.stat().st_size}},
    }}
    new_experiment = {**experiment, "config": extended, "deadline": deadline + 120,
                      "code_revision": "test-continuation", "continuation": {"parent_generation": 1}}
    destination = tmp_path / "extended"
    runner = FakeRunner(extended, destination, fail_candidate=2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(runner, data, destination, "main", 3, deadline + 120, new_experiment)
    runner.log.close()
    forked = load_checkpoint(destination / "checkpoint.pt")
    for key in ("master", "memory", "rng_state", "data_cursor", "metrics", "responses", "generated_tokens"):
        assert equal_state(before[key], forked[key]), key
    audit = json.loads((destination / "fork-audit.json").read_text())
    assert all(audit["preserved_exactly"].values())
    assert forked["generation"] == 1 and forked["generations"] == 3
    assert forked["deadline"] == deadline + 120
    assert file_hash(source) == parent_hash and load_checkpoint(source)["deadline"] == deadline
    with pytest.raises(FileExistsError):
        fork_checkpoint(destination, "main", deadline + 120, new_experiment)
    runner = FakeRunner(extended, destination)
    train(runner, data, destination, "main", 3, deadline + 120, new_experiment)
    actual = load_checkpoint(destination / "checkpoint.pt")
    expected = load_checkpoint(continuous / "checkpoint.pt")
    for key in ("master", "memory", "generation", "data_cursor", "responses", "generated_tokens"):
        assert equal_state(actual[key], expected[key]), key
    assert all(torch.equal(torch.tensor(a["candidate_weights"]), torch.tensor(b["candidate_weights"]))
               for a, b in zip(actual["metrics"], expected["metrics"], strict=True))
    evaluated = evaluate(runner, data, extended, destination / "parent_main", "parent_main", deadline + 120, True,
                         splits=("fresh",))
    assert set(evaluated["sets"]) == {"fresh"} and evaluated["complete"]
    runner.log.close()
    changed = {**extended, "alpha": extended["alpha"] * 2}
    with pytest.raises(ValueError, match="configuration changed"):
        fork_checkpoint(tmp_path / "bad-config", "main", deadline + 120, {**new_experiment, "config": changed})
    with pytest.raises(ValueError, match="absolute deadline"):
        train(FakeRunner(extended, destination), data, destination, "main", 3, deadline + 120,
              {**new_experiment, "deadline": deadline + 121})


def test_parent_guard_rejects_mutation_and_overwrite(tmp_path):
    """A new clock cannot be introduced by overwriting or silently changing its parent."""
    parent = tmp_path / "parent"
    config = {"alpha": 0.1}
    state = {"phase": "finished", "config": config,
             "conditions": {name: {"complete": True, "generation": 19} for name in ("main", "reward_only")}}
    path = parent / "experiment.json"
    atomic_json(path, state)
    child = {**config, "continuation": {"parent_directory": str(parent), "parent_generation": 19,
             "target_generation": 100, "parent_files": {
                 "experiment.json": {"sha256": file_hash(path), "bytes": path.stat().st_size}}}}
    assert validate_parent(child, tmp_path / "child") == state
    with pytest.raises(ValueError, match="separate directory"):
        validate_parent(child, parent / "child")
    with pytest.raises(ValueError, match="training configuration"):
        validate_parent({**child, "alpha": 0.2}, tmp_path / "child")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Parent evidence changed"):
        validate_parent(child, tmp_path / "child")
