"""Protect the primary endpoint's completeness gate using small persisted artifacts."""

import json

import pytest

from novelty_es.reporting import diagnostic_table, summarize


def _evaluation(complete: bool) -> dict:
    """Return explicit synthetic observations, never a model-performance fixture."""
    rows = [
        {
            "problem_id": f"test-{i}",
            "correct": bool(i),
            "valid": True,
            "reward": float(i),
            "generated_tokens": 5,
            "truncated": False,
            "structure": "(+ 1 2)",
        }
        for i in range(2)
    ]
    return {"training_complete": complete, "seconds": 1.0, "sets": {"test": rows}}


def test_primary_requires_completed_training_and_aligned_test_problems(tmp_path):
    """Complete test coverage from an unfinished checkpoint is not the primary result."""
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    config["bootstrap_replicates"] = 100
    experiment = {"config": config, "dataset_manifest": {"splits": {"test": {"count": 2}, "hard": {"count": 1}}}}
    (tmp_path / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    for name in ("main", "reward_only"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "evaluation.json").write_text(json.dumps(_evaluation(name != "main")), encoding="utf-8")
    summary = summarize(tmp_path)
    assert summary["primary"] is None
    assert not summary["methods"]["main"]["complete"]
    assert diagnostic_table(summary)[1][1] == "missing"

    (tmp_path / "main/evaluation.json").write_text(json.dumps(_evaluation(True)), encoding="utf-8")
    summary = summarize(tmp_path)
    assert summary["primary"]["difference"] == 0.0
    assert summary["primary"]["n"] == 2
    assert "does not prove" in summary["interval_caveat"]
    assert not summary["methods"]["main"]["complete"]  # Secondary evaluations remain missing.

    bad = _evaluation(True)
    bad["sets"]["test"][0]["problem_id"] = "unmatched"
    (tmp_path / "main/evaluation.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="do not align"):
        summarize(tmp_path)
