"""Audit a completed real pilot without executing or modifying either language model.

Verify the published data hashes, regrade recorded responses against their original
problems, and compare BF16 exports with final FP32 masters. This complements the
generation-zero audits; it does not replay the optimization trajectory.
Safe tensor API: https://huggingface.co/docs/safetensors/index .
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from novelty_es.continuation import experiment_data, validate_parent
from novelty_es.data import grade, make_prompt
from novelty_es.evaluation import evaluation_sets
from novelty_es.optimizer import candidate_weights, perturbation_seeds
from novelty_es.runtime import atomic_json, load_checkpoint
from novelty_es.schedule import PromptSchedule


def audit_training_trace(directory, data, config, generations, start_generation=0):
    """Check actual logged prompts and noise seeds against the common frozen schedule."""
    schedule = PromptSchedule(len(data["train"]), config["seed"], config["probe_count"], config["rotating_count"])
    expected = {
        (generation, candidate): ([data["train"][i]["id"] for i in schedule.indices(generation)], seed)
        for generation in range(start_generation, generations)
        for candidate, seed in enumerate(perturbation_seeds(config["seed"], generation, config["population"]))
    }
    problems = {row["id"]: row for row in data["train"]}
    prompts, seeds = {name: {} for name in ("main", "reward_only")}, {name: {} for name in ("main", "reward_only")}
    with (directory / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            condition = event.get("condition")
            if event.get("phase") != "train" or condition not in prompts or event["event"] not in ("response", "candidate"):
                continue
            key = (event["generation"], event["candidate"])
            assert key in expected, "Unexpected training candidate"
            if event["event"] == "candidate":
                assert event["seed"] == expected[key][1], "Perturbation seed differs from frozen schedule"
                seeds[condition][key] = event["seed"]
            else:
                problem = problems[event["problem_id"]]
                assert event["prompt"] == make_prompt(problem), "Training prompt differs from original problem"
                assert all(event[field] == value for field, value in asdict(grade(event["response"], problem["numbers"], problem["target"])).items())
                # Replayed rows may recur in a trace; count each scheduled response once.
                prompts[condition].setdefault(key, {})[problem["id"]] = event["prompt"]
    for condition in prompts:
        assert set(prompts[condition]) == set(seeds[condition]) == set(expected)
        assert all(list(prompts[condition][key]) == expected[key][0] for key in expected), "Recorded training prompt order differs"
    assert prompts["main"] == prompts["reward_only"] and seeds["main"] == seeds["reward_only"]
    return {"all_training_prompts_and_seeds_match": True, "all_logged_training_grades_match": True, "candidates_per_arm": len(expected), "scheduled_responses_per_arm": sum(len(ids) for ids, _ in expected.values())}


def main():
    """Require completed endpoints and stop on any mismatch before artifact publication."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/pilot"))
    args = parser.parse_args()
    directory = args.run_dir
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["phase"] == "finished", "Wait for all model execution to finish"
    assert experiment["finished_at"] <= experiment["deadline"], "Model execution exceeded the absolute deadline"
    config = experiment["config"]
    data, manifest = experiment_data(config, Path("data"))
    assert manifest == experiment["dataset_manifest"]
    if "continuation" in experiment:
        validate_parent(config, directory)
    problems = {row["id"]: row for rows in data.values() for row in rows}
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "absolute_deadline": experiment["deadline"], "conditions": {}}
    expected_sets = evaluation_sets(data, config)
    result["paired_training_schedule"] = audit_training_trace(
        directory, data, config, experiment["calibration"]["generations"],
        experiment.get("continuation", {}).get("parent_generation", 0),
    )
    torch.set_num_threads(1)
    conditions = ["original", "main", "reward_only"]
    if "continuation" in experiment:
        conditions.extend(["parent_main", "parent_reward_only"])
    for condition in conditions:
        folder = directory / condition
        evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        assert evaluation["complete"], f"Incomplete evaluation: {condition}"
        assert set(evaluation["sets"]) == ({"fresh"} if condition.startswith("parent_") else set(expected_sets))
        checked = {}
        for split, rows in evaluation["sets"].items():
            for row, scheduled in zip(rows, expected_sets[split], strict=True):
                assert row["problem_id"] == scheduled["id"] and row["sample"] == scheduled.get("sample")
                problem = problems[row["problem_id"]]
                assert row["numbers"] == problem["numbers"] and row["target"] == problem["target"]
                assert row["prompt"] == make_prompt(problem)
                expected = asdict(grade(row["response"], problem["numbers"], problem["target"]))
                assert all(row[key] == value for key, value in expected.items()), f"Grade mismatch: {condition}/{row['problem_id']}"
            checked[split] = len(rows)
        item = {"regraded_responses": checked, "all_grades_match": True}
        if condition in ("main", "reward_only"):
            state = load_checkpoint(folder / "checkpoint.pt")
            if "continuation" in experiment:
                initial = json.loads((folder / "fork-audit.json").read_text(encoding="utf-8"))
                assert all(initial["preserved_exactly"].values())
                assert state["continuation_parent"] == initial["parent"]
                assert initial["parent"]["sha256"] == config["continuation"]["parent_files"][f"{condition}/checkpoint.pt"]["sha256"]
                parent_metrics = json.loads((Path(config["continuation"]["parent_directory"]) / condition / "metrics.json").read_text(encoding="utf-8"))
                assert state["metrics"][:len(parent_metrics)] == parent_metrics
            else:
                initial = json.loads((folder / "initialization-audit.json").read_text(encoding="utf-8"))
                assert initial["all_pretrained_tensors_exactly_equal_in_fp32"] and initial["initial_archive_empty"]
                assert initial["generation"] == 0 and initial["training_code_revision"] == state["code_revision"]
                assert initial["weighting"] == config["weighting"] and initial["novelty_weight"] == config["novelty_weight"]
            assert state["code_revision"] == experiment.get("training_code_revision", experiment["code_revision"])
            assert state["generation"] == state["generations"] == experiment["calibration"]["generations"]
            assert state["config"] == config and state["absolute_deadline"] == experiment["deadline"]
            assert state["deadline"] == experiment["phases"][condition]["deadline"]
            assert state["dataset_manifest"] == experiment["dataset_manifest"]
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            assert metrics == state["metrics"] and len(metrics) == state["generation"]
            assert state["responses"] == sum(row["responses"] for row in metrics)
            assert state["generated_tokens"] == sum(row["generated_tokens"] for row in metrics)
            for row in metrics:
                expected = candidate_weights(row["candidate_quality"], row["candidate_novelty"], config["novelty_weight"] if condition == "main" else 0.0)
                assert torch.equal(torch.tensor(row["candidate_weights"]), expected), "Recorded update coefficients differ"
            with safe_open(folder / "model/model.safetensors", framework="pt", device="cpu") as exported:
                assert set(exported.keys()) == set(state["master"]), "Exported unique parameter names differ"
                for name, tensor in state["master"].items():
                    actual = exported.get_tensor(name)
                    assert actual.dtype == torch.bfloat16 and torch.equal(actual, tensor.to(torch.bfloat16)), f"Export mismatch: {name}"
            item.update(generation=state["generation"], training_code_revision=state["code_revision"], unique_tensors=len(state["master"]), all_exported_tensors_match_final_master=True, all_recorded_update_coefficients_match=True)
            del state
        result["conditions"][condition] = item
    atomic_json(directory / "final-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
