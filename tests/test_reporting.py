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


def test_missing_evaluation_does_not_hide_an_incomplete_condition(tmp_path):
    """A failed arm remains visible even when it never reaches held-out evaluation."""
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    experiment = {
        "config": config,
        "dataset_manifest": {"splits": {"test": {"count": 2}, "hard": {"count": 1}}},
        "conditions": {"reward_only": {"complete": False, "generation": 2, "generations": 5}},
    }
    (tmp_path / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    summary = summarize(tmp_path)
    assert set(summary["methods"]) == {"original", "main", "reward_only"}
    method = summary["methods"]["reward_only"]
    assert not method["training_complete"] and not method["complete"]
    assert method["training"]["generation"] == 2
    assert method["test"]["n"] == 0 and method["test"]["expected"] == 2
    assert summary["primary"] is None


def test_baseline_comparisons_pair_problem_ids_and_require_complete_training(tmp_path):
    """Secondary differences use paired outcomes, never independent error-bar subtraction."""
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    experiment = {"config": config, "dataset_manifest": {"splits": {"test": {"count": 2}, "hard": {"count": 1}}}}
    (tmp_path / "experiment.json").write_text(json.dumps(experiment))
    for name in ("original", "main", "reward_only"):
        (tmp_path / name).mkdir()
        evaluation = _evaluation(name != "reward_only")
        if name == "original":
            evaluation["sets"]["test"].reverse()
        (tmp_path / name / "evaluation.json").write_text(json.dumps(evaluation))
    summary = summarize(tmp_path)
    assert summary["primary"] is None
    assert set(summary["baseline_comparisons"]) == {"main"}
    assert summary["baseline_comparisons"]["main"]["difference"] == 0.0
    assert summary["baseline_comparisons"]["main"]["n"] == 2
