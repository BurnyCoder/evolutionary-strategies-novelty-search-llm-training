"""Explicit registered recovery of an expired attempt into a separate directory.

The expired evidence remains immutable. Only clock/configuration metadata changes;
all learned state and partially evaluated candidates are copied exactly. Serialization:
https://docs.pytorch.org/docs/stable/notes/serialization.html .
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import torch

from .continuation import equal_state, file_hash, training_config
from .runtime import atomic_json, check_budget, load_checkpoint


def recovery_files(source: Path) -> dict:
    """Pin only research evidence, excluding credentials, caches and process logs."""
    patterns = ("experiment.json", "events.jsonl", "*/checkpoint.pt", "*/pending.pt",
                "*/metrics.json", "*/progress.json", "*/fork-audit.json", "original/evaluation.json")
    return {path.relative_to(source).as_posix(): {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in sorted({p for pattern in patterns for p in source.glob(pattern) if p.is_file()})}


def _write_preserved_state(path: Path, state: dict) -> None:
    """Serialize the saved RNGs themselves, rather than capture this preparation process.

    Flush before same-directory atomic replacement, following runtime.save_checkpoint's
    file protocol: https://docs.python.org/3.12/library/os.html#os.replace .
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def initialize_recovery(config: dict, directory: Path, revision: str, now: float | None = None) -> dict:
    """Prepare a distinct allowance once; repeated calls retain its original start/deadline."""
    spec = config["recovery"]
    source, target = Path(spec["source_directory"]).resolve(), directory.resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Recovery must use a separate nonnested directory")
    previous = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    if training_config(config) != training_config(previous["config"]):
        raise ValueError("Recovery cannot change numerical configuration")
    if config.get("continuation") != previous["config"].get("continuation"):
        raise ValueError("Recovery cannot change target, data, parent checkpoints or fresh endpoint")
    if previous["phase"] != "reward_only" or not previous["conditions"]["main"]["complete"]:
        raise ValueError("This recovery requires completed main training and interrupted control")
    state_path = directory / "experiment.json"
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing["registered_config"] != config:
            raise ValueError("Recovery configuration changed on resume")
        return existing
    if recovery_files(source) != spec["files"]:
        raise ValueError("Interrupted source evidence changed")
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "restart-preparation.json"
    if marker.exists():
        clock = json.loads(marker.read_text(encoding="utf-8"))
        if clock["config"] != config:
            raise ValueError("Recovery preparation configuration changed")
    else:
        start = time.time() if now is None else now
        if start <= previous["deadline"]:
            raise ValueError("Use normal resume while the existing allowance is active")
        clock = {"started_at": start, "deadline": min(start + config["total_seconds"], spec["week_deadline"]),
                 "config": config, "code_revision": revision}
        check_budget(clock["deadline"], start)
        atomic_json(marker, clock)
    # Every source file is preserved verbatim alongside the active recovered state.
    for relative, expected in spec["files"].items():
        frozen = directory / "restart-parent" / relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, frozen)
        if file_hash(frozen) != expected["sha256"]:
            raise RuntimeError(f"Recovery evidence copy failed: {relative}")
        if relative != "experiment.json":
            active = directory / relative
            active.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(frozen, active)
    state = copy.deepcopy(previous)
    state.update(started_at=clock["started_at"], deadline=clock["deadline"], config=config,
                 registered_config=config, code_revision=clock["code_revision"],
                 training_code_revision=previous.get("training_code_revision", previous["code_revision"]))
    state["phases"]["reward_only"] = {
        "started_at": clock["started_at"],
        "deadline": min(clock["started_at"] + config["train_seconds"], clock["deadline"]),
    }
    checks = {}
    for name in ("main", "reward_only"):
        for filename in ("checkpoint.pt", "pending.pt"):
            path = directory / name / filename
            if not path.exists():
                continue
            before = load_checkpoint(directory / "restart-parent" / name / filename)
            changed = {"config": config, "deadline": state["phases"][name]["deadline"]}
            if filename == "checkpoint.pt":
                changed["absolute_deadline"] = state["deadline"]
            _write_preserved_state(path, {**before, **changed})
            after = load_checkpoint(path)
            checks[f"{name}/{filename}"] = {key: equal_state(value, after[key])
                                             for key, value in before.items() if key not in changed}
            if not all(checks[f"{name}/{filename}"].values()):
                raise RuntimeError("Recovery changed numerical or random state")
            del before, after
        progress = directory / name / "progress.json"
        if progress.exists():
            row = json.loads(progress.read_text(encoding="utf-8"))
            row["deadline"] = state["phases"][name]["deadline"]
            atomic_json(progress, row)
    state["recovery"] = {
        **spec, "source_status": "interrupted; original allowance expired; paired endpoint incomplete",
        "preserved_exactly": checks, "source_phases": previous["phases"],
        "source_started_at": previous["started_at"], "source_deadline": previous["deadline"],
        "new_started_at": clock["started_at"], "new_deadline": clock["deadline"],
        "recovery_code_revision": clock["code_revision"],
    }
    atomic_json(directory / "restart-audit.json", state["recovery"])
    atomic_json(state_path, state)
    return state
