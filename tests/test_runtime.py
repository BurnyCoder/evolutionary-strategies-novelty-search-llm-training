"""Atomic persistence, deadline integrity and replay equivalence without an LLM."""

import json
import random

import numpy as np
import pytest
import torch

from novelty_es import runtime
from novelty_es.novelty import CorrectMemory
from novelty_es.optimizer import FullWeightES, perturbation_seeds
from novelty_es.runtime import BudgetExpired, check_budget, load_checkpoint, save_checkpoint
from novelty_es.schedule import PromptSchedule, freeze_generation_count


def test_deadline_does_not_reset(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"deadline": 100.0, "generation": 2})
    restored = load_checkpoint(path)
    assert restored["deadline"] == 100.0
    with pytest.raises(BudgetExpired):
        check_budget(restored["deadline"], now=101.0)
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_replace_keeps_last_good_checkpoint_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"generation": 1})

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_checkpoint(path, {"generation": 2})
    assert load_checkpoint(path)["generation"] == 1


def test_atomic_json_retries_temporary_windows_reader_lock(tmp_path, monkeypatch):
    """A reader may briefly deny replacement; retry without exposing half a JSON file."""
    path = tmp_path / "evaluation.json"
    runtime.atomic_json(path, {"deadline": 100, "completed": 16})
    replace, calls = runtime.os.replace, []

    def temporarily_locked(source, destination):
        calls.append(destination)
        if len(calls) < 3:
            assert json.loads(path.read_text())["completed"] == 16
            error = PermissionError("simulated Windows sharing conflict")
            error.winerror = 5
            raise error
        return replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", temporarily_locked)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    runtime.atomic_json(path, {"deadline": 100, "completed": 32})
    assert len(calls) == 3
    assert json.loads(path.read_text()) == {"deadline": 100, "completed": 32}
    assert not path.with_suffix(".json.tmp").exists()


def test_atomic_json_persistent_lock_preserves_previous_record(tmp_path, monkeypatch):
    """Bounded retries must still surface a persistent storage failure."""
    path = tmp_path / "evaluation.json"
    runtime.atomic_json(path, {"completed": 16})

    def locked(*args):
        error = PermissionError("persistent sharing conflict")
        error.winerror = 32
        raise error

    monkeypatch.setattr(runtime.os, "replace", locked)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        runtime.atomic_json(path, {"completed": 32})
    assert json.loads(path.read_text()) == {"completed": 16}


def test_schedule_no_probe_leakage_and_random_access():
    schedule = PromptSchedule(103, seed=0, probe_count=16, rotating_count=32)
    for generation in range(10):
        batch = schedule.indices(generation)
        assert len(batch) == len(set(batch)) == 48
        assert batch[-16:] == schedule.probes
        assert batch == PromptSchedule(103, 0, 16, 32).indices(generation)


def test_freeze_uses_slowest_candidate_and_overheads():
    count, seconds = freeze_generation_count(
        [1.0, 2.0, 3.0], 4.0, 5.0, 10.0, population=30, validation_every=5, allowance=16200, fraction=0.8, cap=500
    )
    assert seconds == 101.0
    assert count == int(12960 / 101)


def test_interrupted_and_continuous_es_equivalent(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    es = FullWeightES(model, 0.001, 0.0005, "cpu")
    memory = CorrectMemory(6, 0)
    embedding = torch.randn(3, 2, 4)
    correct = torch.ones(3, 2, dtype=torch.bool)
    es.update(perturbation_seeds(0, 0, 3), [1.0, 0.1, 0.5])
    memory.append(embedding, correct)
    path = tmp_path / "state.pt"
    save_checkpoint(path, {"master": es.master, "memory": memory.state_dict(), "deadline": 900.0})
    expected_rng = (random.random(), np.random.rand(), torch.rand(1))
    es.update(perturbation_seeds(0, 1, 3), [0.5, 0.2, 0.8])
    memory.append(embedding, correct)
    restored = load_checkpoint(path, restore_rng=True)
    actual_rng = (random.random(), np.random.rand(), torch.rand(1))
    assert expected_rng[:2] == actual_rng[:2]
    assert torch.equal(expected_rng[2], actual_rng[2])
    other = FullWeightES(torch.nn.Linear(3, 2), 0.001, 0.0005, "cpu")
    other.load_master(restored["master"])
    other_memory = CorrectMemory(6, 0)
    other_memory.load_state_dict(restored["memory"])
    other.update(perturbation_seeds(0, 1, 3), [0.5, 0.2, 0.8])
    other_memory.append(embedding, correct)
    assert all(torch.equal(es.master[k], other.master[k]) for k in es.master)
    assert all(torch.equal(memory.history[k], other_memory.history[k]) for k in memory.history)
