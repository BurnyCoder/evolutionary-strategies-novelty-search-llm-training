"""Fork a completed registered experiment without resetting its historical allowance.

Only orchestration metadata changes. FP32 parameters, archive, random streams and
global schedules are inherited exactly. PyTorch checkpoint loading uses weights_only:
https://docs.pytorch.org/docs/stable/notes/serialization.html .
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from .data import load_data
from .runtime import atomic_json, load_checkpoint, save_checkpoint


def file_hash(path: Path) -> str:
    """Hash immutable parent evidence without loading multi-gigabyte tensors into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_config(config: dict) -> dict:
    """Exclude only explicitly permitted follow-up bookkeeping, never training settings."""
    metadata = {
        "experiment", "train_seconds", "final_seconds", "total_seconds", "primary_split",
        "publication_suffix", "artifact_repo", "continuation",
    }
    return {key: value for key, value in config.items() if key not in metadata}


def experiment_data(config: dict, directory: Path) -> tuple[dict, dict]:
    """Validate pinned data and merge an optional fresh evaluation split with no leakage."""
    data = load_data(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if "continuation" not in config:
        return data, manifest
    spec = config["continuation"]
    fresh_directory = Path(spec["fresh_directory"])
    fresh_manifest = json.loads((fresh_directory / "manifest.json").read_text(encoding="utf-8"))
    if fresh_manifest != spec["fresh_manifest"] or fresh_manifest["excluded_source_manifest"] != manifest:
        raise ValueError("Fresh evaluation manifest or exclusion source changed")
    fresh = load_data(fresh_directory)
    seen = {tuple(sorted(row["numbers"])) for rows in data.values() for row in rows}
    for rows in fresh.values():
        for row in rows:
            group = tuple(sorted(row["numbers"]))
            if group in seen:
                raise ValueError("Fresh evaluation multiset overlaps an earlier problem")
            seen.add(group)
    if data.keys() & fresh.keys():
        raise ValueError("Fresh split would overwrite an existing split")
    return {**data, **fresh}, {
        **manifest, "splits": {**manifest["splits"], **fresh_manifest["splits"]},
        "fresh_source_manifest": fresh_manifest,
    }


def validate_parent(config: dict, destination: Path) -> dict:
    """Require completed, pinned source files and unchanged numerical configuration."""
    spec = config["continuation"]
    parent = Path(spec["parent_directory"]).resolve()
    target = destination.resolve()
    if target == parent or target.is_relative_to(parent) or parent.is_relative_to(target):
        raise ValueError("Continuation must use a separate directory outside its parent")
    for relative, expected in spec["parent_files"].items():
        path = parent / relative
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Parent evidence changed: {relative}")
    previous = json.loads((parent / "experiment.json").read_text(encoding="utf-8"))
    if previous["phase"] != "finished":
        raise ValueError("Finish the parent experiment before extending it")
    if training_config(previous["config"]) != training_config(config):
        raise ValueError("Continuation must preserve the registered training configuration")
    for name in ("main", "reward_only"):
        status = previous["conditions"][name]
        if not status["complete"] or status["generation"] != spec["parent_generation"]:
            raise ValueError("Both parent conditions must have the registered completed generation")
    if spec["target_generation"] <= spec["parent_generation"]:
        raise ValueError("Continuation target must add generations")
    return previous


def initialize_continuation(state: dict, directory: Path, parent: dict) -> None:
    """Set a new, bounded experiment phase while retaining all parent deadline provenance."""
    config = state["config"]
    spec = config["continuation"]
    state["deadline"] = min(state["deadline"], spec["pretrained_allocation_deadline"])
    if state["deadline"] <= state["started_at"]:
        raise ValueError("The original pretrained-study allocation has expired")
    state["phase"], state["phases"] = "main", {}
    state["continuation"] = {
        "parent_generation": spec["parent_generation"],
        "target_generation": spec["target_generation"],
        "parent_started_at": parent["started_at"],
        "parent_deadline": parent["deadline"],
        "parent_code_revision": parent.get("training_code_revision", parent["code_revision"]),
        "parent_files": spec["parent_files"],
    }
    state["calibration"] = {
        "generations": spec["target_generation"], "batch_size": config["batch_size"],
        "source": "registered parent timing; no new calibration or weight updates",
        **spec["timing_rule"],
    }
    # Reuse only unchanged original-model outcomes; fresh responses remain unobserved.
    source = Path(spec["parent_directory"]) / "original/evaluation.json"
    baseline = json.loads(source.read_text(encoding="utf-8"))
    baseline["reused_from"] = {"path": source.as_posix(), "sha256": file_hash(source), "seconds": baseline["seconds"]}
    baseline["complete"] = False
    baseline["seconds"] = 0.0
    atomic_json(directory / "original/evaluation.json", baseline)


def equal_state(first, second) -> bool:
    """Compare nested tensor/primitive state exactly for fork and replay audits."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            equal_state(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(
            equal_state(a, b) for a, b in zip(first, second, strict=True)
        )
    return first == second


def fork_checkpoint(directory: Path, condition: str, deadline: float, experiment: dict) -> dict:
    """Copy complete state at the phase boundary, verifying every preserved field.

    Restoring before saving prevents save_checkpoint from capturing unrelated RNGs.
    No random operation occurs between source restoration and the atomic fork save.
    """
    config = experiment["config"]
    spec = config["continuation"]
    source = Path(spec["parent_directory"]) / condition / "checkpoint.pt"
    expected = spec["parent_files"][f"{condition}/checkpoint.pt"]
    if file_hash(source) != expected["sha256"]:
        raise ValueError("Parent checkpoint hash changed")
    parent = load_checkpoint(source, restore_rng=True)
    if parent["condition"] != condition or parent["generation"] != spec["parent_generation"]:
        raise ValueError("Wrong parent condition or generation")
    if training_config(parent["config"]) != training_config(config):
        raise ValueError("Checkpoint training configuration changed")
    manifest = copy.deepcopy(experiment["dataset_manifest"])
    fresh_manifest = manifest.pop("fresh_source_manifest", None)
    if fresh_manifest:
        for split in fresh_manifest["splits"]:
            manifest["splits"].pop(split)
    if parent["dataset_manifest"] != manifest:
        raise ValueError("Parent training data changed")
    checkpoint = directory / "checkpoint.pt"
    lineage = {
        "source": source.as_posix(), "sha256": expected["sha256"],
        "generation": parent["generation"], "code_revision": parent["code_revision"],
        "deadline": parent["deadline"], "absolute_deadline": parent["absolute_deadline"],
    }
    forked = {
        **parent, "config": config, "deadline": deadline, "absolute_deadline": experiment["deadline"],
        "generations": spec["target_generation"], "dataset_manifest": experiment["dataset_manifest"],
        "code_revision": experiment["code_revision"], "continuation_parent": lineage,
    }
    if checkpoint.exists():
        # A crash after saving the fork but before its audit may finish verification,
        # without replacing the checkpoint or granting a new deadline.
        loaded = load_checkpoint(checkpoint)
        if (directory / "fork-audit.json").exists() or any(
            not equal_state(loaded.get(key), forked[key])
            for key in ("generation", "generations", "config", "deadline", "absolute_deadline", "continuation_parent")
        ):
            raise FileExistsError("Refuse to overwrite an existing continuation checkpoint")
    else:
        save_checkpoint(checkpoint, forked)
        loaded = load_checkpoint(checkpoint)
    preserved = ["master", "memory", "rng_state", "generation", "data_cursor", "generated_tokens", "responses", "metrics"]
    checks = {key: equal_state(parent[key], loaded[key]) for key in preserved}
    if not all(checks.values()):
        raise RuntimeError("Continuation fork failed exact state equivalence")
    audit = {"parent": lineage, "preserved_exactly": checks, "new_deadline": deadline,
             "new_absolute_deadline": experiment["deadline"], "new_code_revision": experiment["code_revision"]}
    atomic_json(directory / "fork-audit.json", audit)
    atomic_json(directory / "metrics.json", parent["metrics"])
    return audit
