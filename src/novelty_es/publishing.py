"""Publish only explicit model/evaluation artifacts into one research collection.

Hub API reference: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api .
The user authorized publishing this research; no credentials or unrelated files are uploaded.
"""

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .runtime import atomic_json


def artifact_manifest(directory: Path) -> dict:
    """Hash the explicit release set after the writer has finished, excluding local extras.

    Streaming hashes keep multi-gigabyte FP32 checkpoints out of memory:
    https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest .
    """
    patterns = [
        "README.md", "LICENSE", "experiment.json", "calibration.json", "summary.json", "final-audit.json",
        "before-amendment-A.json", "config-amendment-A.json", "recovery-audit.json",
        "failure-evaluation-uncommitted.json", "events.jsonl", "*/checkpoint.pt",
        "*/evaluation.json", "*/metrics.json", "*/progress.json", "*/initialization-audit.json", "*/fork-audit.json",
    ]
    files = {}
    for path in sorted({path for pattern in patterns for path in directory.glob(pattern) if path.is_file()}):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[path.relative_to(directory).as_posix()] = {"sha256": digest, "bytes": path.stat().st_size}
    return {"algorithm": "sha256", "files": files, "note": "The manifest excludes itself and the later upload receipt."}


def publish(directory: Path) -> dict:
    """Upload completed local exports and raw evaluation evidence, preserving attribution."""
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if experiment["phase"] != "finished" or not (directory / "summary.json").exists():
        raise RuntimeError("Generate the first experiment report before publishing models")
    config = experiment["config"]
    if "continuation" in config and (
        config.get("publication_suffix", "seed0") == "seed0"
        or config.get("artifact_repo", "novelty-es-pilot-artifacts") == "novelty-es-pilot-artifacts"
    ):
        raise ValueError("Continuation publication must use distinct model and evidence repositories")
    api = HfApi()
    namespace = api.whoami()["name"]
    collection = api.create_collection(
        title="Evolution Strategies with LLM Novelty",
        namespace=namespace,
        description="Preregistered full-weight ES, correctness-gated novelty, controls and reports.",
        exists_ok=True,
    )
    published = {"collection": collection.url, "models": {}, "artifacts": None, "revisions": {}}
    for condition in ("main", "reward_only"):
        model_dir = directory / condition / "model"
        if not model_dir.exists():
            continue
        # Preserve the pinned base model's redistribution license in each derivative.
        config = experiment["config"]
        license_path = hf_hub_download(config["model_id"], "LICENSE", revision=config["model_revision"])
        shutil.copyfile(license_path, model_dir / "LICENSE")
        suffix = config.get("publication_suffix", "seed0")
        repo_id = f"{namespace}/novelty-es-qwen2.5-0.5b-{condition.replace('_', '-')}-{suffix}"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            folder_path=model_dir,
            commit_message=f"Publish registered {condition} {config.get('experiment', 'pilot')} checkpoint and model card",
        )
        api.add_collection_item(collection.slug, repo_id, "model", exists_ok=True)
        published["models"][condition] = f"https://huggingface.co/{repo_id}"
        published["revisions"][condition] = commit.oid
        atomic_json(directory / "published.json", published)
    repo_id = f"{namespace}/{experiment['config'].get('artifact_repo', 'novelty-es-pilot-artifacts')}"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    manifest = artifact_manifest(directory)
    atomic_json(directory / "artifact-manifest.json", manifest)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=directory,
        allow_patterns=[*manifest["files"], "artifact-manifest.json"],
        commit_message="Publish raw evaluations, traces and experiment provenance",
    )
    api.add_collection_item(collection.slug, repo_id, "dataset", exists_ok=True)
    published["artifacts"] = f"https://huggingface.co/datasets/{repo_id}"
    published["revisions"]["artifacts"] = commit.oid
    atomic_json(directory / "published.json", published)
    print(json.dumps(published, indent=2))
    return published
