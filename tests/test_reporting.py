"""Protect the primary endpoint's completeness gate using small persisted artifacts."""

import json

import pytest

from novelty_es.continuation import experiment_data
from novelty_es.reporting import diagnostic_table, report, result_table, summarize
from novelty_es.runtime import atomic_json


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
    (tmp_path / "reward_only").mkdir()
    (tmp_path / "reward_only/metrics.json").write_text(json.dumps([{
        "responses": 2, "generated_tokens": 10, "truncated": 0,
        "candidate_quality": [0.0, 1.0], "candidate_novelty": [0.0, 0.5],
    }]))
    summary = summarize(tmp_path)
    assert set(summary["methods"]) == {"original", "main", "reward_only"}
    method = summary["methods"]["reward_only"]
    assert not method["training_complete"] and not method["complete"]
    assert method["training"]["generation"] == 2
    assert method["test"]["n"] == 0 and method["test"]["expected"] == 2
    diagnostic = method["training_observations"]["weight_diagnostic"]
    assert diagnostic["candidates_positive_novelty"] == 1
    assert diagnostic["candidates_weight_changed"] == 0  # Control computes novelty but never applies it.
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


def test_fresh_primary_and_paired_change_require_the_fresh_endpoint(tmp_path):
    """Old-test completeness cannot satisfy the new primary; changes pair all four outcomes."""
    config = json.loads(open("configs/pilot.json", encoding="utf-8").read())
    config.update(primary_split="fresh", bootstrap_replicates=100)
    experiment = {"config": config, "continuation": {"parent_generation": 19, "target_generation": 100},
                  "dataset_manifest": {"splits": {"test": {"count": 2}, "hard": {"count": 1},
                                                   "fresh": {"count": 2}}}}
    atomic_json(tmp_path / "experiment.json", experiment)
    for name in ("original", "main", "reward_only"):
        atomic_json(tmp_path / name / "evaluation.json", _evaluation(True))
    assert summarize(tmp_path)["primary"] is None
    values = {"original": [False, False], "main": [True, True], "reward_only": [False, True],
              "parent_main": [False, True], "parent_reward_only": [False, True]}
    for name, correctness in values.items():
        evaluation = _evaluation(True)
        evaluation["sets"]["fresh"] = [
            {**row, "problem_id": f"fresh-{i}", "correct": correctness[i]}
            for i, row in enumerate(evaluation["sets"]["test"])
        ]
        if name.startswith("parent_"):
            evaluation["sets"].pop("test")
        atomic_json(tmp_path / name / "evaluation.json", evaluation)
    summary = summarize(tmp_path)
    assert summary["primary_split"] == "fresh"
    assert summary["primary"]["difference"] == 0.5
    assert summary["longitudinal"]["difference_in_differences"]["difference"] == 0.5
    assert summary["longitudinal"]["changes"]["main"]["difference"] == 0.5
    assert summary["longitudinal"]["old_test_difference"]["difference"] == 0
    assert summary["discordance"] == {"both_correct": 1, "main_only": 1, "control_only": 0, "both_wrong": 0}


def test_continuation_report_keeps_the_pilot_release_separate(tmp_path):
    """Exercise the complete report entry point without any language-model execution."""
    from pathlib import Path

    config = json.loads(Path("configs/continuation.json").read_text())
    config["bootstrap_replicates"] = 100
    _, manifest = experiment_data(config, Path("data"))
    directory, output = tmp_path / "run", tmp_path / "release"
    atomic_json(directory / "experiment.json", {
        "config": config, "dataset_manifest": manifest, "phase": "main",
        "continuation": {"parent_generation": 19, "target_generation": 100},
    })
    preserved = output / "reports/experiment-pilot.md"
    preserved.parent.mkdir(parents=True)
    preserved.write_text("immutable original report", encoding="utf-8")
    before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    summary = report(directory, output, staged=True)
    assert before == {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    assert (output / 'releases/continuation-seed0-gen100/summary.json').exists()
    assert json.loads((output / 'docs/site/continuation-seed0-gen100/status.json').read_text())['summary'] == summary
    assert summary["primary"] is None
    assert preserved.read_text() == "immutable original report"
    assert (output / "reports/experiment-continuation-seed0-gen100.md").is_file()
    assert (output / "output/pdf/novelty-es-continuation-seed0-gen100.pdf").stat().st_size > 1000
    assert (output / "docs/site/continuation-seed0-gen100/status.json").is_file()
    assert not (output / "docs/site/status.json").exists()
    # Regenerating the pilot after the continuation must retain both studies.
    pilot = tmp_path / 'pilot'
    pilot_config = json.loads(Path('configs/pilot.json').read_text())
    pilot_config['bootstrap_replicates'] = 100
    _, pilot_manifest = experiment_data(pilot_config, Path('data'))
    atomic_json(pilot / 'experiment.json', {'config': pilot_config, 'dataset_manifest': pilot_manifest})
    report(pilot, output, staged=True)
    aggregate = (output / 'reports/all-experiments.md').read_text()
    assert 'experiment-pilot.md' in aggregate
    assert 'experiment-continuation-seed0-gen100.md' in aggregate
    assert 'Original-test pass@8' in aggregate


def test_sampled_indices_are_checked_even_in_incomplete_groups(tmp_path):
    config = json.loads(open('configs/pilot.json', encoding='utf-8').read())
    atomic_json(tmp_path / 'experiment.json', {
        'config': config, 'dataset_manifest': {'splits': {'test': {'count': 2}, 'hard': {'count': 1}}},
    })
    evaluation = _evaluation(True)
    row = {**evaluation['sets']['test'][0], 'sample': 0}
    evaluation['sets']['sampled'] = [row, row]
    atomic_json(tmp_path / 'main/evaluation.json', evaluation)
    with pytest.raises(ValueError, match='sample'):
        summarize(tmp_path)
    evaluation['sets']['sampled'] = [{**row, 'sample': config['samples_per_problem']}]
    atomic_json(tmp_path / 'main/evaluation.json', evaluation)
    with pytest.raises(ValueError, match='sample'):
        summarize(tmp_path)


def test_result_headers_separate_greedy_and_sampled_splits():
    header = result_table({'primary_split': 'fresh', 'methods': {}})[0]
    assert 'Fresh greedy accuracy' in header
    assert 'Original-test pass@8' in header


def test_recovery_report_discloses_expired_attempt(tmp_path):
    """The new result cannot present the missed original deadline as fulfilled."""
    from pathlib import Path

    config = json.loads(Path("configs/continuation-recovery.json").read_text())
    config["bootstrap_replicates"] = 100
    _, manifest = experiment_data(config, Path("data"))
    directory, output = tmp_path / "run", tmp_path / "release"
    recovery = {**config["recovery"], "source_phases": {"reward_only": {"started_at": 1794074074.0}}}
    atomic_json(directory / "experiment.json", {
        "config": config, "dataset_manifest": manifest, "phase": "reward_only", "recovery": recovery,
        "continuation": {"parent_generation": 19, "target_generation": 100},
    })
    summary = report(directory, output)
    assert summary["primary"] is None
    report_text = (output / "reports/experiment-continuation-seed0-gen100.md").read_text(encoding="utf-8")
    assert "that attempt remains incomplete" in report_text
    assert "preregistration-recovery.md" in report_text
    assert (output / "reports/experiment-continuation-interrupted.md").exists()
