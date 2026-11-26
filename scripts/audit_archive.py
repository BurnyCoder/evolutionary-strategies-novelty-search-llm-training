"""Reconstruct archive *identities* from correct probe events, and distances where tensors survive.

The unit of identity is a specific observed response, never an embedding guessed from
another occurrence of the same text. No embedding model or optimizer is instantiated.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from novelty_es.verification import read, separate_output


def saved(path):
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def scan(directory, first, last, config, probes):
    groups = defaultdict(dict)
    seeds = {}
    for line in (directory / "events.jsonl").open("rb"):
        if b'"train"' not in line or (
            b'"seed"' not in line and b'"correct": true' not in line and b'"correct":true' not in line
        ):
            continue
        e = json.loads(line)
        if e.get("phase") != "train":
            continue
        arm, g, c = e["condition"], e["generation"], e["candidate"]
        if e["event"] == "candidate":
            if seeds.setdefault((arm, g, c), e["seed"]) != e["seed"]:
                raise ValueError("Replayed candidate changed seed")
        elif e["event"] == "response" and e["correct"] and e["problem_id"] in probes:
            p = probes[e["problem_id"]]
            value = (e["problem_id"], e["response"], e["token_ids"])
            if groups[arm, g, p].setdefault(c, value) != value:
                raise ValueError("Replayed correct probe changed")
    expected = {}
    for g in range(first, last):
        population = random.Random(f"es-v1:{config['seed']}:{g}").sample(range(2**62), config["population"])
        for arm in ("main", "reward_only"):
            expected.update({(arm, g, c): seed for c, seed in enumerate(population)})
    if expected != seeds:
        raise ValueError("Missing or unequal matched candidate seeds")
    return groups, len(seeds)


def check_arm(arm, groups, config, metrics, checkpoints, pending, parent):
    rng = torch.Generator().manual_seed(config["seed"])
    size = config["memory_size"]
    histories = [[] for _ in range(config["probe_count"])]
    vectors, snapshots, requests = {}, {}, {}
    for path in pending:
        state = saved(path)
        g = state["generation"]
        for c, entry in enumerate(state["candidates"]):
            for p, on in enumerate(entry["correct"].tolist()):
                row = entry["records"][config["rotating_count"] + p]
                actual = groups.get((arm, g, p), {}).get(c)
                if bool(actual) != on or (on and (row["problem_id"], row["response"], row["token_ids"]) != actual):
                    raise ValueError("Pending probe mask or raw occurrence changed")
                if on:
                    vectors[g, p, c] = normalize(entry["embeddings"][p].float(), dim=0)
    for g in range(len(metrics)):
        for p, history in enumerate(histories):
            current = [(g, p, c) for c in sorted(groups.get((arm, g, p), {}))]
            for k in current:
                requests[k] = history + [other for other in current if other != k]
            if current:
                order = torch.randperm(len(current), generator=rng).tolist()
                histories[p] = (history + [current[i] for i in order])[-size:]
        for path in checkpoints.get(g + 1, []):
            state = saved(path)
            memory = state["memory"]
            if memory["capacity"] != size or not torch.equal(memory["rng"], rng.get_state()):
                raise ValueError("Archive insertion RNG differs from reconstructed stream")
            if set(memory["history"]) != {p for p, h in enumerate(histories) if h}:
                raise ValueError("Archive probe keys differ")
            paired = 0
            for p, tensors in memory["history"].items():
                for k, v in zip(histories[p], tensors, strict=True):
                    if k in vectors:
                        if not torch.allclose(v, vectors[k], rtol=0, atol=2e-6):
                            raise ValueError("Retained archive vector differs from same identified occurrence")
                        paired += 1
                    vectors[k] = v.clone()
            snapshots[str(path.relative_to(path.parents[2]))] = dict(
                generation=g + 1,
                archive_rows=sum(map(len, histories)),
                overlapping_vector_comparisons=paired,
                insertion_rng_matches=True,
            )
    result = {}
    for label, lo, hi in (("pilot", 0, parent), ("added", parent, len(metrics))):
        count = Counter()
        largest = 0.0
        for g in range(lo, hi):
            for c, observed in enumerate(metrics[g]["candidate_novelty"]):
                bonuses = []
                active = False
                for p in range(config["probe_count"]):
                    k = g, p, c
                    if k not in requests:
                        count["incorrect_gate_zero"] += 1
                        bonuses.append(0.0)
                        continue
                    active = True
                    refs = requests[k]
                    if not refs:
                        count["correct_empty_pool"] += 1
                        bonuses.append(1.0)
                    elif k in vectors and all(r in vectors for r in refs):
                        nearest = (torch.stack([vectors[r] for r in refs]) @ vectors[k]).max().clamp(-1, 1)
                        bonuses.append(1 - float(nearest))
                        count["correct_with_tensor_distance"] += 1
                    else:
                        count["correct_missing_distance_inputs"] += 1
                if len(bonuses) == config["probe_count"]:
                    count["fully_checkable_candidate_means"] += 1
                    count["fully_checkable_with_correct_probe"] += int(active)
                    error = abs(float(np.mean(bonuses)) - observed)
                    if error > 2e-6:
                        raise ValueError(f"Saved novelty mean differs at {arm}/{g}/{c}")
                    largest = max(largest, error)
        result[label] = dict(**count, max_abs_mean_error=largest)
    return dict(snapshots=snapshots, retained_identified_vectors=len(vectors), stages=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot-dir", "continuation-dir", "data-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    separate_output(a.output.parent, [a.pilot_dir, a.continuation_dir, a.data_dir])
    torch.set_num_threads(1)
    cfg = read(a.continuation_dir / "experiment.json")["config"]
    parent, end = cfg["continuation"]["parent_generation"], cfg["continuation"]["target_generation"]
    rows = [json.loads(line)["id"] for line in (a.data_dir / "train.jsonl").read_bytes().splitlines()]
    random.Random(f"probes-v1:{cfg['seed']}").shuffle(rows)
    probes = {pid: i for i, pid in enumerate(rows[: cfg["probe_count"]])}
    groups, n = scan(a.pilot_dir, 0, parent, cfg, probes)
    added, m = scan(a.continuation_dir, parent, end, cfg, probes)
    groups.update(added)
    result = dict(pilot_candidate_seed_matches=n, added_candidate_seed_matches=m, arms={})
    for arm in ("main", "reward_only"):
        source = a.continuation_dir / "restart-parent" / arm
        paths = defaultdict(list)
        for path in (
            a.pilot_dir / arm / "checkpoint.pt",
            source / "checkpoint.pt",
            a.continuation_dir / arm / "checkpoint.pt",
        ):
            paths[saved(path)["generation"]].append(path)
        result["arms"][arm] = check_arm(
            arm, groups, cfg, read(a.continuation_dir / arm / "metrics.json"), paths, [source / "pending.pt"], parent
        )
    a.output.parent.mkdir(exist_ok=True, parents=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
