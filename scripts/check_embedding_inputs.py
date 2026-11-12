"""Measure existing probe strings against the pinned tokenizer cap; no model inference.

Tokenizer API: https://huggingface.co/docs/transformers/main_classes/tokenizer .
The cap is read from the pinned sentence-transformers configuration, not inferred
from the causal model's different token budget. Output must be outside evidence.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from novelty_es.verification import read, separate_output, sha256


def check(directory: Path) -> dict:
    """Tokenize logged fixed probes, retaining replayed records as separate observations."""
    config = read(directory / "experiment.json")["config"]
    options = dict(repo_id=config["embedding_id"], revision=config["embedding_revision"], local_files_only=True)
    cap_path = Path(hf_hub_download(filename="sentence_bert_config.json", **options))
    cap = read(cap_path)["max_seq_length"]
    tokenizer = AutoTokenizer.from_pretrained(config["embedding_id"], revision=config["embedding_revision"],
                                              local_files_only=True, trust_remote_code=False)
    result = dict(logged_probe_responses=0, above_limit=0, correct_above_limit=0, max_tokens=0,
                  limit=cap, embedding_revision=config["embedding_revision"], config_sha256=sha256(cap_path))
    with (directory / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "response" or row.get("phase") != "train" or row["offset"] < config["rotating_count"]:
                continue
            length = len(tokenizer(row["response"], truncation=False, add_special_tokens=True)["input_ids"])
            result["logged_probe_responses"] += 1
            result["above_limit"] += length > cap
            result["correct_above_limit"] += length > cap and row["correct"]
            result["max_tokens"] = max(result["max_tokens"], length)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--continuation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    separate_output(args.output.parent, [args.pilot_dir, args.continuation_dir])
    result = {name: check(path) for name, path in (("pilot", args.pilot_dir), ("continuation", args.continuation_dir))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
