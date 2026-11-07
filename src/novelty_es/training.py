"""Resumable population evaluation and bounded full-weight ES training phases."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .models import ModelRunner
from .novelty import CorrectMemory
from .optimizer import FullWeightES, perturbation_seeds
from .runtime import atomic_json, check_budget, load_checkpoint, save_checkpoint
from .schedule import PromptSchedule, freeze_generation_count


def response_metrics(records: list[dict]) -> dict:
    """Measure task quantities directly from exact verifier outcomes, with denominators."""
    count = len(records)
    return {
        "accuracy": sum(row["correct"] for row in records) / count,
        "quality": sum(row["reward"] for row in records) / count,
        "validity": sum(row["valid"] for row in records) / count,
        "generated_tokens": sum(row["generated_tokens"] for row in records),
        "truncated": sum(row["truncated"] for row in records),
        "responses": count,
    }


def generate_rows(runner, rows: list[dict], deadline: float, scope: dict) -> list[dict]:
    """Chunk a fixed schedule without changing which prompts each candidate receives."""
    records = []
    for offset in range(0, len(rows), runner.config["batch_size"]):
        records.extend(
            runner.generate(rows[offset : offset + runner.config["batch_size"]], deadline, {**scope, "offset": offset})
        )
    return records


def candidate(runner, optimizer, rows: list[dict], seed: int, deadline: float, scope: dict) -> dict:
    """Rebuild one candidate and compute identical task/embedding work for both arms."""
    check_budget(deadline)
    tick = time.perf_counter()
    optimizer.reconstruct(seed)
    records = generate_rows(runner, rows, deadline, scope)
    probe_records = records[-runner.config["probe_count"] :]
    embeddings = runner.embed([row["response"] for row in probe_records])
    correct = torch.tensor([row["correct"] for row in probe_records], dtype=torch.bool)
    result = {
        "seed": seed,
        "records": records,
        "embeddings": embeddings,
        "correct": correct,
        "seconds": time.perf_counter() - tick,
        **response_metrics(records),
    }
    runner.log.emit(
        "candidate",
        **scope,
        **{
            key: result[key]
            for key in ("seed", "seconds", "accuracy", "quality", "validity", "generated_tokens", "truncated")
        },
    )
    return result


def calibrate(runner, data: dict, directory: Path, deadline: float) -> dict:
    """Measure only registered workload components; no metric can tune the budget."""
    config = runner.config
    optimizer = FullWeightES(runner.model, config["sigma"], config["alpha"], runner.device)
    schedule = PromptSchedule(len(data["train"]), config["seed"], config["probe_count"], config["rotating_count"])
    rows = [data["train"][i] for i in schedule.indices(0)]
    seeds = perturbation_seeds(config["seed"], 0, config["population"])
    runner.load_embedder()
    # OOM fallback is mechanical and frozen before the timing measurements.
    while True:
        try:
            optimizer.reconstruct(seeds[0])
            generate_rows(
                runner,
                rows[: config["batch_size"]],
                deadline,
                {"phase": "calibration_warmup", "condition": "calibration"},
            )
            break
        except torch.OutOfMemoryError:
            runner.log.emit("oom_retry", batch_size=config["batch_size"])
            if config["batch_size"] == 1:
                raise
            config["batch_size"] //= 2
            torch.cuda.empty_cache()
    timings = []
    for index in range(config["calibration_candidates"]):
        measured = candidate(
            runner,
            optimizer,
            rows,
            seeds[index],
            deadline,
            {"phase": "calibration", "condition": "calibration", "candidate": index},
        )
        timings.append(measured["seconds"])
    check_budget(deadline)
    tick = time.perf_counter()
    # Synthetic nonflat ranks force the complete 30-stream update cost, never fit labels.
    optimizer.update(seeds, list(range(config["population"])))
    update_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    save_checkpoint(directory / "calibration.pt", {"master": optimizer.master, "deadline": deadline})
    checkpoint_seconds = time.perf_counter() - tick
    optimizer.reconstruct(None)
    tick = time.perf_counter()
    generate_rows(
        runner,
        data["validation"][: config["validation_count"]],
        deadline,
        {"phase": "calibration_validation", "condition": "calibration"},
    )
    validation_seconds = time.perf_counter() - tick
    count, estimate = freeze_generation_count(
        timings,
        update_seconds,
        checkpoint_seconds,
        validation_seconds,
        config["population"],
        config["validation_every"],
        config["train_seconds"],
        config["timing_fraction"],
        config["max_generations"],
    )
    result = {
        "generations": count,
        "estimated_generation_seconds": estimate,
        "candidate_seconds": timings,
        "update_seconds": update_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "validation_seconds": validation_seconds,
        "batch_size": config["batch_size"],
        "frozen_at": time.time(),
    }
    atomic_json(directory / "calibration.json", result)
    runner.log.emit("calibration_frozen", **result)
    return result


def train(
    runner: ModelRunner,
    data: dict,
    directory: Path,
    condition: str,
    generations: int,
    deadline: float,
    experiment: dict,
) -> dict:
    """Resume the exact next candidate or generation and preserve the original deadline."""
    config = runner.config
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, pending_path = directory / "checkpoint.pt", directory / "pending.pt"
    optimizer = FullWeightES(runner.model, config["sigma"], config["alpha"], runner.device)
    memory = CorrectMemory(config["memory_size"], config["seed"])
    schedule = PromptSchedule(len(data["train"]), config["seed"], config["probe_count"], config["rotating_count"])
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, restore_rng=True)
        if state["config"] != config or state["deadline"] != deadline or state["generations"] != generations:
            raise ValueError("Cannot change registered configuration or allowance on resume")
        if state["dataset_manifest"] != experiment["dataset_manifest"]:
            raise ValueError("Checkpoint dataset differs")
        optimizer.load_master(state["master"])
        memory.load_state_dict(state["memory"])
    else:
        state = {
            "generation": 0,
            "generations": generations,
            "condition": condition,
            "config": config,
            "deadline": deadline,
            "absolute_deadline": experiment["deadline"],
            "dataset_manifest": experiment["dataset_manifest"],
            "code_revision": experiment["code_revision"],
            "data_cursor": 0,
            "generated_tokens": 0,
            "responses": 0,
            "metrics": [],
            "master": optimizer.master,
            "memory": memory.state_dict(),
        }
        save_checkpoint(checkpoint, state)
    runner.load_embedder()
    while state["generation"] < generations:
        check_budget(deadline)
        generation = state["generation"]
        tick = time.perf_counter()
        rows = [data["train"][i] for i in schedule.indices(generation)]
        seeds = perturbation_seeds(config["seed"], generation, config["population"])
        pending = load_checkpoint(pending_path) if pending_path.exists() else {}
        if pending.get("generation") == generation:
            pending = load_checkpoint(pending_path, restore_rng=True)
        if pending.get("generation") != generation:
            pending = {
                "generation": generation,
                "candidates": [],
                "config": config,
                "deadline": deadline,
                "condition": condition,
                "seconds": 0.0,
            }
        if pending["config"] != config or pending["deadline"] != deadline or pending["condition"] != condition:
            raise ValueError("Pending generation metadata differs")
        candidates = pending["candidates"]
        if [entry["seed"] for entry in candidates] != seeds[: len(candidates)]:
            raise ValueError("Pending candidate seed schedule differs")
        for index in range(len(candidates), config["population"]):
            entry = candidate(
                runner,
                optimizer,
                rows,
                seeds[index],
                deadline,
                {"phase": "train", "condition": condition, "generation": generation, "candidate": index},
            )
            candidates.append(entry)
            pending["seconds"] += entry["seconds"]
            save_checkpoint(pending_path, pending)
            atomic_json(
                directory / "progress.json",
                {
                    "condition": condition,
                    "generation": generation,
                    "generations": generations,
                    "candidate": index + 1,
                    "population": config["population"],
                    "deadline": deadline,
                    "checkpoint_generation": state["generation"],
                    **{key: entry[key] for key in ("accuracy", "quality", "validity", "generated_tokens")},
                },
            )
        embeddings = torch.stack([row["embeddings"] for row in candidates])
        correct = torch.stack([row["correct"] for row in candidates])
        novelty = memory.score(embeddings, correct).mean(dim=1)
        quality = torch.tensor([row["quality"] for row in candidates])
        fitness = quality + (config["novelty_weight"] if condition == "main" else 0.0) * novelty
        check_budget(deadline)
        optimizer.update(seeds, fitness.tolist())
        memory.append(embeddings, correct)
        optimizer.reconstruct(None)
        metrics = {
            "generation": generation + 1,
            "quality": float(quality.mean()),
            "novelty": float(novelty.mean()),
            "fitness": float(fitness.mean()),
            "accuracy": sum(row["accuracy"] for row in candidates) / len(candidates),
            "validity": sum(row["validity"] for row in candidates) / len(candidates),
            "generated_tokens": sum(row["generated_tokens"] for row in candidates),
            "responses": sum(row["responses"] for row in candidates),
            "truncated": sum(row["truncated"] for row in candidates),
            "candidate_quality": quality.tolist(),
            "candidate_novelty": novelty.tolist(),
        }
        if (generation + 1) % config["validation_every"] == 0 or generation + 1 == generations:
            validation = generate_rows(
                runner,
                data["validation"][: config["validation_count"]],
                deadline,
                {"phase": "validation", "condition": condition, "generation": generation + 1},
            )
            metrics["validation"] = response_metrics(validation)
        check_budget(deadline)
        # Commit generation state only after scoring, update and scheduled validation finish.
        # Include already completed candidates after resume, not just this process's work.
        metrics["seconds"] = max(time.perf_counter() - tick, pending["seconds"])
        metrics["candidate_seconds_total"] = pending["seconds"]
        state["generation"] += 1
        state["data_cursor"] = state["generation"] * config["rotating_count"]
        state["generated_tokens"] += metrics["generated_tokens"]
        state["responses"] += metrics["responses"]
        state["metrics"].append(metrics)
        state["master"], state["memory"] = optimizer.master, memory.state_dict()
        save_checkpoint(checkpoint, state)
        atomic_json(directory / "metrics.json", state["metrics"])
        atomic_json(
            directory / "progress.json",
            {
                "condition": condition,
                "generation": state["generation"],
                "generations": generations,
                "candidate": config["population"],
                "population": config["population"],
                "deadline": deadline,
                "checkpoint_generation": state["generation"],
                **metrics,
            },
        )
        runner.log.emit("generation_committed", condition=condition, **metrics)
    return {
        "generation": state["generation"],
        "generations": generations,
        "complete": True,
        "generated_tokens": state["generated_tokens"],
        "responses": state["responses"],
    }
