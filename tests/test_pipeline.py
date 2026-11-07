"""Exercise real training/evaluation persistence with a deterministic CPU stand-in.

This test emits synthetic responses from known witnesses and must never be
reported as language-model performance. It checks orchestration, not capability.
"""

import json
import time
from dataclasses import asdict

import pytest
import torch

from novelty_es.data import generate_dataset, grade, make_prompt
from novelty_es.evaluation import evaluate
from novelty_es.runtime import EventLog, load_checkpoint
from novelty_es.training import train


class FakeRunner:
    """Tiny weight-sensitive response generator for interruption and update tests."""

    def __init__(self, config, directory, fail_candidate=None):
        self.config, self.device = config, "cpu"
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.model.weight.data.zero_()
        self.log = EventLog(directory / "fake-events.jsonl")
        self.fail_candidate = fail_candidate

    def load_embedder(self):
        pass

    def embed(self, texts):
        return torch.tensor([[len(text) + 1.0, sum(map(ord, text)) % 17 + 1.0, 1.0] for text in texts])

    def generate(self, rows, deadline, scope, **kwargs):
        if scope.get("candidate") == self.fail_candidate and self.fail_candidate is not None:
            raise RuntimeError("simulated interruption")
        records = []
        for row in rows:
            response = row["witness"] if self.model.weight.sum().item() > 0 else "+".join(map(str, row["numbers"]))
            records.append(
                {
                    "problem_id": row["id"],
                    "prompt": make_prompt(row),
                    "response": response,
                    "generated_tokens": len(response),
                    "truncated": False,
                    "batch_seconds": 0.01,
                    **asdict(grade(response, row["numbers"], row["target"])),
                }
            )
        return records


def test_population_interruption_replay_and_evaluation(tmp_path):
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    config.update(
        population=4,
        probe_count=2,
        rotating_count=4,
        validation_count=4,
        validation_every=1,
        batch_size=2,
        sampled_problem_count=4,
    )
    data = generate_dataset(5, {"train": 20, "validation": 6, "test": 8, "hard": 4})
    deadline = time.time() + 120
    experiment = {"deadline": deadline, "dataset_manifest": {"synthetic_test": True}, "code_revision": "test"}
    uninterrupted = FakeRunner(config, tmp_path / "continuous")
    train(uninterrupted, data, tmp_path / "continuous", "main", 2, deadline, experiment)
    interrupted = FakeRunner(config, tmp_path / "resume", fail_candidate=2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(interrupted, data, tmp_path / "resume", "main", 2, deadline, experiment)
    assert len(load_checkpoint(tmp_path / "resume/pending.pt")["candidates"]) == 2
    resumed = FakeRunner(config, tmp_path / "resume")
    train(resumed, data, tmp_path / "resume", "main", 2, deadline, experiment)
    a = load_checkpoint(tmp_path / "continuous/checkpoint.pt")
    b = load_checkpoint(tmp_path / "resume/checkpoint.pt")
    assert a["generation"] == b["generation"] == 2
    assert a["deadline"] == b["deadline"] == deadline
    assert all(torch.equal(a["master"][key], b["master"][key]) for key in a["master"])
    assert all(torch.equal(a["memory"]["history"][key], b["memory"]["history"][key]) for key in a["memory"]["history"])
    result = evaluate(resumed, data, config, tmp_path / "resume", "main", deadline, True)
    assert result["complete"]
    assert len(result["sets"]["sampled"]) == 4 * 16
    assert evaluate(resumed, data, config, tmp_path / "resume", "main", deadline, True) == result
