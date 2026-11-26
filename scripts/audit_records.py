"""Rebuild the registered-rule audit from immutable inputs, using CPU records only."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from novelty_es.evidence_trace import inspect_trace
from novelty_es.scientific_audit import (
    evaluation_statistics,
    grade_registered,
    interval,
    paired_result,
    require_endpoint,
)
from novelty_es.verification import datasets, read, scheduled_rows, separate_output, sha256


def inspect_checkpoint(path, target):
    """Inspect saved tensors and archive contents, never constructing a neural module."""
    state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if state["generation"] != target:
        raise ValueError(f"Unexpected checkpoint generation at {path}")
    capacity = state["memory"]["capacity"]
    history = state["memory"]["history"]
    errors = []
    for probe, values in history.items():
        if not (
            0 <= probe < state["config"]["probe_count"]
            and 0 < len(values) <= capacity
            and values.ndim == 2
            and values.shape[1] == 384
            and torch.isfinite(values).all()
        ):
            raise ValueError("Malformed archived embeddings")
        errors.append(float((values.norm(dim=-1) - 1).abs().max()))
    if max(errors, default=0) > 1e-5:
        raise ValueError("Archived embeddings are not normalized")
    if state["data_cursor"] != target * state["config"]["rotating_count"] or len(state["metrics"]) != target:
        raise ValueError("Checkpoint cursor or metrics do not match its generation")
    return dict(
        generation=target,
        archive_capacity=capacity,
        archive_rows={str(k): len(v) for k, v in history.items()},
        max_archive_unit_norm_error=max(errors, default=0),
        saved_rng_fields=[k for k in state if "rng" in k],
        saved_archive_rng_bytes=state["memory"]["rng"].numel(),
        code_revision=state["code_revision"],
        data_cursor=state["data_cursor"],
        master_tensor_count=len(state["master"]),
    )


def analyze(directory, data, name):
    """Pair final and parent outcomes only after verifying their identity and target."""
    exp = read(directory / "experiment.json")
    config = exp["config"]
    split = config.get("primary_split", "test")
    target = config.get("continuation", {}).get("target_generation", exp["calibration"]["generations"])
    scheduled = scheduled_rows(data, config)
    arms = ["original", "main", "reward_only"] + (["parent_main", "parent_reward_only"] if split == "fresh" else [])
    differences, raw, results, flags, hashes, checkpoints = [], {}, {}, {}, {}, {}
    for arm in arms:
        path = directory / arm / "evaluation.json"
        hashes[f"{arm}/evaluation.json"] = sha256(path)
        raw[arm] = evaluation = read(path)
        expected = {"fresh": scheduled["fresh"]} if arm.startswith("parent_") else scheduled
        if (
            evaluation["condition"] != arm
            or not evaluation["training_complete"]
            or not evaluation["complete"]
            or expected.keys() != evaluation["sets"].keys()
        ):
            raise ValueError(f"Incomplete or misidentified {arm} evaluation")
        if arm in ("main", "reward_only"):
            require_endpoint(exp["conditions"][arm], target)
            checkpoints[arm] = inspect_checkpoint(directory / arm / "checkpoint.pt", target)
        results[arm], flags[arm] = {}, {}
        for key, rows in evaluation["sets"].items():
            results[arm][key], flags[arm][key] = evaluation_statistics(
                rows, expected[key], key, config, differences, arm
            )
    reps = config["bootstrap_replicates"]
    comparisons = {"primary": paired_result(flags["main"][split], flags["reward_only"][split], reps)}
    for arm in ("main", "reward_only"):
        comparisons[f"{arm}_minus_original"] = paired_result(flags[arm][split], flags["original"][split], reps)
    if split == "fresh":
        changes = {}
        for arm in ("main", "reward_only"):
            parent = flags[f"parent_{arm}"]["fresh"]
            comparisons[f"{arm}_change"] = paired_result(flags[arm][split], parent, reps)
            changes[arm] = {k: int(v) - int(parent[k]) for k, v in flags[arm][split].items()}
        comparisons["difference_in_changes"] = interval(
            [changes["main"][k] - changes["reward_only"][k] for k in sorted(changes["main"])], reps
        )
        comparisons["old_test"] = paired_result(flags["main"]["test"], flags["reward_only"]["test"], reps)
    for relative in ("experiment.json", "events.jsonl", "main/metrics.json", "reward_only/metrics.json"):
        hashes[relative] = sha256(directory / relative)
    print(f"Reconciling {name} prompts, tokens and candidate records", flush=True)
    trace = inspect_trace(directory, data, config, raw)
    if split == "fresh":
        first = min(
            datetime.fromisoformat(t["first"]).timestamp()
            for k, t in trace["phase_prompt_times"].items()
            if k.startswith("evaluation/") and k.endswith("/fresh")
        )
        if first <= max(exp["phases"][arm]["finished_at"] for arm in ("main", "reward_only")):
            raise ValueError("Fresh evaluation preceded end of training")
    return dict(
        primary_split=split,
        target=target,
        input_hashes=hashes,
        checkpoints=checkpoints,
        trace=trace,
        evaluations=results,
        comparisons=comparisons,
        evaluation_discrepancies=differences,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("pilot-dir", "continuation-dir", "data-dir", "fresh-dir", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    dirs = [args.pilot_dir, args.continuation_dir, args.data_dir, args.fresh_dir]
    separate_output(args.output.parent, dirs)
    torch.set_num_threads(1)
    data, hashes = datasets(dirs[2:])
    for split, rows in data.items():
        lengths = [len(x["numbers"]) for x in rows]
        if (split == "hard" and set(lengths) != {5}) or (
            split != "hard" and not lengths.count(3) == lengths.count(4) == len(rows) // 2
        ):
            raise ValueError("Registered number-count balance differs")
        for row in rows:
            if not (
                all(1 <= n <= 50 for n in row["numbers"])
                and 1 <= row["target"] <= 999
                and grade_registered(row["witness"], row["numbers"], row["target"])[1]
            ):
                raise ValueError(f"Invalid problem witness {row['id']}")
    evidence = dict(revision_date="2026-11-26", dataset_hashes=hashes, studies={})
    for name, directory in zip(("pilot", "continuation"), dirs, strict=False):
        study_data = data if name == "continuation" else {k: v for k, v in data.items() if k != "fresh"}
        evidence["studies"][name] = analyze(directory, study_data, name)
    deadline = read(args.continuation_dir / "restart-parent/experiment.json")
    recovered = read(args.continuation_dir / "experiment.json")
    before, after = (
        torch.load(args.continuation_dir / p, weights_only=True, map_location="cpu", mmap=True)
        for p in ("restart-parent/main/checkpoint.pt", "main/checkpoint.pt")
    )
    if before["master"].keys() != after["master"].keys() or any(
        not torch.equal(v, after["master"][k]) for k, v in before["master"].items()
    ):
        raise ValueError("Recovered main changed")
    original = read(args.pilot_dir / "original/evaluation.json")["sets"]
    reused = read(args.continuation_dir / "original/evaluation.json")["sets"]
    if any(v != reused[k] for k, v in original.items()):
        raise ValueError("Reused pilot rows changed")
    evidence["recovery"] = dict(
        expired_deadline=deadline["deadline"],
        recovery_start=recovered["started_at"],
        source_checkpoint_generations=recovered["recovery"]["source_generations"],
        preserved_main_tensors=len(before["master"]),
        reused_original_splits=list(original),
    )
    if not math.isfinite(deadline["deadline"]) or recovered["started_at"] <= deadline["deadline"]:
        raise ValueError("Unexpected recovery chronology")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
