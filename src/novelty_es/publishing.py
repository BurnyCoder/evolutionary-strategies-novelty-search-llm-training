"""Publish only explicit model/evaluation artifacts into one research collection.

Hub API reference: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api .
The user authorized publishing this research; no credentials or unrelated files are uploaded.
"""

import json
from pathlib import Path

from huggingface_hub import HfApi

from .runtime import atomic_json


def publish(directory: Path) -> dict:
    """Upload completed local exports and raw evaluation evidence, preserving attribution."""
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if experiment["phase"] != "finished" or not (directory / "summary.json").exists():
        raise RuntimeError("Generate the first experiment report before publishing models")
    api = HfApi()
    namespace = api.whoami()["name"]
    collection = api.create_collection(
        title="Evolution Strategies with LLM Novelty",
        namespace=namespace,
        description="Preregistered full-weight ES, correctness-gated novelty, controls and reports.",
        exists_ok=True,
    )
    published = {"collection": collection.url, "models": {}, "artifacts": None}
    for condition in ("main", "reward_only"):
        model_dir = directory / condition / "model"
        if not model_dir.exists():
            continue
        repo_id = f"{namespace}/novelty-es-qwen2.5-0.5b-{condition.replace('_', '-')}-seed0"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=model_dir,
            commit_message=f"Publish registered {condition} pilot checkpoint and model card",
        )
        api.add_collection_item(collection.slug, repo_id, "model", exists_ok=True)
        published["models"][condition] = f"https://huggingface.co/{repo_id}"
    repo_id = f"{namespace}/novelty-es-pilot-artifacts"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=directory,
        allow_patterns=[
            "experiment.json",
            "calibration.json",
            "summary.json",
            "events.jsonl",
            "*/evaluation.json",
            "*/metrics.json",
            "*/progress.json",
        ],
        commit_message="Publish raw pilot evaluations, traces and experiment provenance",
    )
    api.add_collection_item(collection.slug, repo_id, "dataset", exists_ok=True)
    published["artifacts"] = f"https://huggingface.co/datasets/{repo_id}"
    atomic_json(directory / "published.json", published)
    print(json.dumps(published, indent=2))
    return published
