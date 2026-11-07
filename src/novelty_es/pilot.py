"""Ordered experiment phases with persistent clocks and no implicit budget extension."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import torch

from .data import load_data
from .evaluation import evaluate
from .models import ModelRunner
from .optimizer import FullWeightES
from .runtime import BudgetExpired, EventLog, atomic_json, check_budget, configure_runtime, load_checkpoint
from .training import calibrate, train


def _git(*args: str) -> str:
    """Read provenance using argument vectors, without shell expansion or credentials."""
    return subprocess.check_output(["git", *args], text=True).strip()


def _environment() -> dict:
    """Record reproducibility metadata, excluding environment variables and secrets."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "sentence-transformers", "numpy", "huggingface-hub")
        },
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def run_pilot(config: dict, directory: Path, data_directory: Path = Path("data")) -> dict:
    """Run/resume calibration, baseline, main, control, final evaluation and reporting."""
    if os.environ.get("NOVELTY_DEVICE", "cuda") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This registered pilot requires the local CUDA GPU")
    directory.mkdir(parents=True, exist_ok=True)
    configure_runtime(config["seed"])
    data = load_data(data_directory)
    manifest = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    state_path = directory / "experiment.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if config != state["registered_config"] or manifest != state["dataset_manifest"]:
            raise ValueError("Registered configuration/data changed on resume")
        config = state["config"]
    else:
        if _git("status", "--porcelain"):
            raise RuntimeError("Publish and commit all core changes before starting the experiment")
        start = time.time()
        state = {
            "started_at": start,
            "deadline": start + config["total_seconds"],
            "phase": "calibration",
            "config": copy.deepcopy(config),
            "registered_config": copy.deepcopy(config),
            "dataset_manifest": manifest,
            "code_revision": _git("rev-parse", "HEAD"),
            "environment": _environment(),
            "phases": {},
            "conditions": {},
            "failures": [],
        }
        # The first durable state creates the clock; subsequent executions never replace it.
        state["phases"]["calibration"] = {"started_at": start, "deadline": start + config["calibration_seconds"]}
        atomic_json(state_path, state)
    log = EventLog(directory / "events.jsonl")
    log.emit(
        "pilot_entered",
        phase=state["phase"],
        absolute_deadline=state["deadline"],
        code_revision=_git("rev-parse", "HEAD"),
    )
    if state["phase"] == "finished":
        log.close()
        return state
    try:
        if state["phase"] == "calibration":
            limit = min(state["deadline"], state["phases"]["calibration"]["deadline"])
            calibration_file = directory / "calibration.json"
            if calibration_file.exists():
                calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
                config["batch_size"] = calibration["batch_size"]
            else:
                check_budget(limit)
                runner = ModelRunner(config, "cuda", log)
                try:
                    calibration = calibrate(runner, data, directory, limit)
                finally:
                    runner.close()
            state["calibration"], state["config"] = calibration, config
            atomic_json(state_path, state)  # Freeze before evaluating original held-out outcomes.
            runner = None
            try:
                check_budget(limit)
                runner = ModelRunner(config, "cuda", log)
                evaluate(runner, data, config, directory / "original", "original", limit, True)
            except BudgetExpired as error:
                log.emit("baseline_deferred", reason=str(error))
            finally:
                if runner:
                    runner.close()
            state["phases"]["calibration"]["finished_at"] = time.time()
            state["phase"] = "main"
            atomic_json(state_path, state)
        for condition in ("main", "reward_only"):
            if state["phase"] != condition:
                continue
            if condition not in state["phases"]:
                now = time.time()
                state["phases"][condition] = {
                    "started_at": now,
                    "deadline": min(now + config["train_seconds"], state["deadline"]),
                }
                atomic_json(state_path, state)
            phase = state["phases"][condition]
            runner = None
            try:
                check_budget(phase["deadline"])
                runner = ModelRunner(config, "cuda", log)
                result = train(
                    runner,
                    data,
                    directory / condition,
                    condition,
                    state["calibration"]["generations"],
                    phase["deadline"],
                    state,
                )
            except BudgetExpired as error:
                checkpoint = directory / condition / "checkpoint.pt"
                completed = load_checkpoint(checkpoint)["generation"] if checkpoint.exists() else 0
                result = {
                    "complete": False,
                    "generation": completed,
                    "generations": state["calibration"]["generations"],
                    "reason": str(error),
                }
                log.emit("training_incomplete", condition=condition, **result)
            finally:
                if runner:
                    runner.close()
            phase["finished_at"] = time.time()
            state["conditions"][condition] = result
            state["phase"] = "reward_only" if condition == "main" else "final"
            atomic_json(state_path, state)
        if state["phase"] == "final":
            if "final" not in state["phases"]:
                now = time.time()
                state["phases"]["final"] = {
                    "started_at": now,
                    "deadline": min(now + config["final_seconds"], state["deadline"]),
                }
                atomic_json(state_path, state)
            limit = state["phases"]["final"]["deadline"]
            for condition in ("main", "reward_only", "original"):
                runner = None
                try:
                    check_budget(limit)
                    complete = condition == "original" or state["conditions"][condition]["complete"]
                    runner = ModelRunner(config, "cuda", log)
                    if condition != "original":
                        checkpoint = directory / condition / "checkpoint.pt"
                        if not checkpoint.exists():
                            log.emit("evaluation_unavailable", condition=condition, reason="no complete checkpoint")
                            continue
                        saved = load_checkpoint(checkpoint)
                        optimizer = FullWeightES(runner.model, config["sigma"], config["alpha"], "cuda")
                        optimizer.load_master(saved["master"])
                        del optimizer, saved
                    evaluate(runner, data, config, directory / condition, condition, limit, complete)
                    # Hugging Face-ready exports are local artifacts; network publishing is separate.
                    if condition != "original":
                        check_budget(limit)
                        runner.model.save_pretrained(directory / condition / "model", safe_serialization=True)
                        runner.tokenizer.save_pretrained(directory / condition / "model")
                except BudgetExpired as error:
                    log.emit("evaluation_incomplete", condition=condition, reason=str(error))
                    break
                finally:
                    if runner:
                        runner.close()
            state["phases"]["final"]["finished_at"] = time.time()
            state["phase"], state["finished_at"] = "finished", time.time()
            atomic_json(state_path, state)
            log.emit("pilot_finished", elapsed_seconds=state["finished_at"] - state["started_at"])
        return state
    except Exception as error:
        state["failures"].append(
            {"time": time.time(), "phase": state["phase"], "exception": type(error).__name__, "message": str(error)}
        )
        atomic_json(state_path, state)
        log.emit("failure", phase=state["phase"], error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        log.close()
