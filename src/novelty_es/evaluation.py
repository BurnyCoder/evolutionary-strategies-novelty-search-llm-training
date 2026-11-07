"""Resumable held-out inference with frozen problem and sampling schedules."""

import json
import random
import time
from pathlib import Path

from .runtime import atomic_json, check_budget
from .schedule import PromptSchedule


def evaluation_sets(data: dict, config: dict) -> dict[str, list[dict]]:
    """Construct each endpoint without exposing any held-out data to optimization."""
    schedule = PromptSchedule(len(data["train"]), config["seed"], config["probe_count"], config["rotating_count"])
    indices = random.Random(config["data_seed"]).sample(range(len(data["test"])), config["sampled_problem_count"])
    samples = []
    for index in indices:
        for sample in range(config["samples_per_problem"]):
            samples.append({**data["test"][index], "sample": sample})
    return {
        **({"fresh": data["fresh"]} if "fresh" in data else {}),
        "test": data["test"],
        "hard": data["hard"],
        "probes": [data["train"][i] for i in schedule.probes],
        "sampled": samples,
    }


def evaluate(
    runner, data: dict, config: dict, directory: Path, condition: str, deadline: float, training_complete: bool,
    splits: tuple[str, ...] | None = None,
) -> dict:
    """Persist whole completed batches; restarting reuses outcomes, never selects answers."""
    path = directory / "evaluation.json"
    state = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "condition": condition,
            "training_complete": training_complete,
            "sets": {},
            "seconds": 0.0,
            "complete": False,
            "model_revision": config["model_revision"],
        }
    )
    if state["training_complete"] != training_complete:
        raise ValueError("Cannot mix outcomes from different checkpoints")
    for name, rows in evaluation_sets(data, config).items():
        if splits is not None and name not in splits:
            continue
        records = state["sets"].setdefault(name, [])
        while len(records) < len(rows):
            check_budget(deadline)
            start = len(records)
            batch = rows[start : start + config["batch_size"]]
            tick = time.perf_counter()
            outputs = runner.generate(
                batch,
                deadline,
                {"phase": "evaluation", "condition": condition, "split": name, "offset": start},
                sampled=name == "sampled",
                sample_seed=config["data_seed"] + start,
            )
            for row, output in zip(batch, outputs, strict=True):
                output["sample"] = row.get("sample")
            records.extend(outputs)
            state["seconds"] += time.perf_counter() - tick
            atomic_json(path, state)
            runner.log.emit(
                "evaluation_progress",
                condition=condition,
                split=name,
                completed=len(records),
                total=len(rows),
                accuracy=sum(x["correct"] for x in records) / len(records),
            )
    state["complete"] = True
    atomic_json(path, state)
    return state
