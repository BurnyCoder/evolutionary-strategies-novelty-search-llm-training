"""Verify a real generation-zero checkpoint against every pinned pretrained tensor.

Run before the first generation commits. This CPU-only audit reads snapshots and
never edits weights, archive or experiment state. It records evidence separately.
Safe tensor access: https://huggingface.co/docs/safetensors/index .
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from safetensors import safe_open

from novelty_es.runtime import atomic_json, load_checkpoint


def main():
    """Refuse later checkpoints so successful training cannot be mistaken for initialization."""
    load_dotenv()
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=["main", "reward_only"])
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/pilot"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = load_checkpoint(args.run_dir / args.condition / "checkpoint.pt")
    assert state["generation"] == 0, "The checkpoint is no longer the initial state"
    assert state["memory"]["history"] == {}, "Initial memory must be empty"
    config = state["config"]
    assert config["weighting"] == "zscore_task_plus_novelty_v1"
    source = hf_hub_download(
        config["model_id"], "model.safetensors", revision=config["model_revision"], local_files_only=True,
    )
    model_config = json.loads(Path(hf_hub_download(
        config["model_id"], "config.json", revision=config["model_revision"], local_files_only=True,
    )).read_text(encoding="utf-8"))
    with safe_open(source, framework="pt", device="cpu") as original:
        assert set(state["master"]) == set(original.keys()), "Unique parameter names differ"
        for name, value in state["master"].items():
            assert torch.equal(value, original.get_tensor(name).float()), f"Changed initial tensor: {name}"
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "condition": args.condition,
        "generation": state["generation"], "model_id": config["model_id"], "model_revision": config["model_revision"],
        "training_code_revision": state["code_revision"], "weighting": config["weighting"],
        "novelty_weight": config["novelty_weight"], "unique_tensors": len(state["master"]),
        "parameter_elements": sum(value.numel() for value in state["master"].values()),
        "architecture": {key: model_config[key] for key in (
            "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "max_position_embeddings", "tie_word_embeddings",
        )},
        "all_pretrained_tensors_exactly_equal_in_fp32": True, "initial_archive_empty": True,
        "absolute_deadline": state["absolute_deadline"],
    }
    atomic_json(args.run_dir / args.condition / "initialization-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
