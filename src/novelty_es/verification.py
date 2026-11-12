"""Read-only reanalysis of persisted evidence, independent of training and old audits.

No model is constructed or executed. Arithmetic uses a separate precedence parser;
Python's parser is used only to enforce the historical accepted syntax and size cap.
https://docs.python.org/3.12/reference/expressions.html
https://docs.python.org/3.12/library/fractions.html
pass@k: https://arxiv.org/abs/2107.03374 (Equation 1).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np


def read(path: Path):
    """Read one JSON artifact without updating it."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """Stream large checkpoints rather than reading their bytes into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def separate_output(output: Path, sources: list[Path]) -> None:
    """Reject overlapping trees so a mistaken output cannot replace source evidence."""
    target = output.resolve()
    for source in sources:
        source = source.resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError(f"Output must be separate from input: {source}")


def inventory(directory: Path) -> dict:
    """Hash evidence and weights; exclude only documented live service bookkeeping."""
    transient = {"dashboard-error.log", "dashboard.log", "awake.log", "writer.lock"}
    return {p.relative_to(directory).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in transient
            and ".cache" not in p.relative_to(directory).parts}


def reference_grade(response: str, numbers: list[int], target: int) -> dict:
    """Independently evaluate allowed tokens with precedence climbing and exact rationals."""
    invalid = dict(reward=0.0, correct=False, valid=False, expression="", structure="")
    tags = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if len(response) > 8192 or len(tags) > 1:
        return invalid
    expression = (tags[0] if tags else response).strip()
    if not re.fullmatch(r"[0-9+*/()\s-]{1,512}", expression):
        return invalid
    tokens = re.findall(r"[0-9]+|[^\s]", expression)
    cursor, literals = 0, []

    def parse(minimum=0):
        nonlocal cursor
        token = tokens[cursor]
        cursor += 1
        if token.isdigit():
            value, structure = Fraction(int(token)), str(int(token))
            literals.append(int(token))
        elif token == "(":
            value, structure = parse()
            if tokens[cursor] != ")":
                raise ValueError("Unclosed parentheses")
            cursor += 1
        else:
            raise ValueError("A positive integer or parenthesized expression is required")
        while cursor < len(tokens):
            op = tokens[cursor]
            priority = {"+": 1, "-": 1, "*": 2, "/": 2}.get(op, -1)
            if priority < minimum:
                break
            cursor += 1
            right, right_structure = parse(priority + 1)
            if op == "+":
                value += right
            elif op == "-":
                value -= right
            elif op == "*":
                value *= right
            else:
                value /= right
            children = sorted((structure, right_structure)) if op in ("+", "*") else (structure, right_structure)
            structure = f"({op} {children[0]} {children[1]})"
        return value, structure

    try:
        if len(list(ast.walk(ast.parse(expression, mode="eval")))) > 64:
            return invalid
        value, structure = parse()
        if cursor != len(tokens) or Counter(literals) != Counter(numbers):
            return invalid
    except (IndexError, ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return invalid
    correct = value == target
    return dict(reward=1.0 if correct else 0.1, correct=correct, valid=True,
                expression=expression, structure=structure)


def datasets(directories: list[Path]) -> tuple[dict, dict]:
    """Check content hashes, counts, witnesses and global multiset separation."""
    data, hashes, groups, ids = {}, {}, set(), set()
    for directory in directories:
        manifest = read(directory / "manifest.json")
        for split, info in manifest["splits"].items():
            path = directory / f"{split}.jsonl"
            if split in data or sha256(path) != info["sha256"]:
                raise ValueError(f"Duplicate split or changed dataset: {split}")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != info["count"]:
                raise ValueError(f"Dataset count differs: {split}")
            for row in rows:
                group = tuple(sorted(row["numbers"]))
                if group in groups or row["id"] in ids:
                    raise ValueError("Repeated dataset identity or number multiset")
                groups.add(group)
                ids.add(row["id"])
                if not reference_grade(row["witness"], row["numbers"], row["target"])["correct"]:
                    raise ValueError(f"Invalid solvability witness: {row['id']}")
            data[split], hashes[split] = rows, info
    return data, hashes


def scheduled_rows(data: dict, config: dict, generation: int | None = None) -> dict | list:
    """Reconstruct the registered schedules without using the experiment's scheduler."""
    order = list(range(len(data["train"])))
    random.Random(f"probes-v1:{config['seed']}").shuffle(order)
    probes, pool = order[:config["probe_count"]], order[config["probe_count"]:]
    if generation is not None:
        epoch, batch = divmod(generation, len(pool) // config["rotating_count"])
        random.Random(f"epoch-v1:{config['seed']}:{epoch}").shuffle(pool)
        start = batch * config["rotating_count"]
        return [data["train"][i] for i in pool[start:start + config["rotating_count"]] + probes]
    sample = random.Random(config["data_seed"]).sample(data["test"], config["sampled_problem_count"])
    return {**{key: data[key] for key in ("test", "hard", "fresh") if key in data},
            "probes": [data["train"][i] for i in probes],
            "sampled": [{**row, "sample": i} for row in sample for i in range(config["samples_per_problem"])]}


def check_row(row: dict, expected: dict) -> None:
    """Verify prompt identity, exact grading and persisted token accounting."""
    for key in ("numbers", "target"):
        if row[key] != expected[key]:
            raise ValueError(f"Problem changed: {row['problem_id']}")
    prompt = (f"Use the numbers {expected['numbers']} exactly once each to make {expected['target']}. "
              "You may use +, -, *, / and parentheses. Do not use other numbers or operations. "
              "Return only <answer>expression</answer>.")
    grade = reference_grade(row["response"], row["numbers"], row["target"])
    if row["prompt"] != prompt or any(row[key] != value for key, value in grade.items()):
        raise ValueError(f"Prompt or independent grade differs: {row['problem_id']}")
    if type(row["correct"]) is not bool or type(row["valid"]) is not bool:
        raise ValueError("Correctness and validity must be explicit Booleans")
    if len(row["token_ids"]) != row["generated_tokens"]:
        raise ValueError("Token count differs from saved token IDs")


def verify_evaluation(evaluation: dict, expected: dict, config: dict) -> dict:
    """Reject missing, duplicated, shuffled and unexpected evaluation records."""
    if set(evaluation["sets"]) != set(expected):
        raise ValueError("Evaluation split coverage differs")
    result = {}
    for split, rows in evaluation["sets"].items():
        schedule = expected[split]
        if len(rows) != len(schedule):
            raise ValueError(f"Incomplete or duplicated evaluation: {split}")
        for row, problem in zip(rows, schedule, strict=True):
            if row["problem_id"] != problem["id"] or row.get("sample") != problem.get("sample"):
                raise ValueError(f"Evaluation identity/sample schedule differs: {split}")
            check_row(row, problem)
        result[split] = {
            "responses": len(rows), "correct": sum(row["correct"] for row in rows),
            "valid": sum(row["valid"] for row in rows), "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
        }
        if split == "sampled":
            n = config["samples_per_problem"]
            counts = [sum(row["correct"] for row in rows[i:i + n]) for i in range(0, len(rows), n)]
            result[split].update({f"pass{k}": float(np.mean([
                1 - math.comb(n - correct, k) / math.comb(n, k) for correct in counts
            ])) for k in (1, 8)})
            result[split]["mean_unique_correct_structures"] = float(np.mean([
                len({row["structure"] for row in rows[i:i + n] if row["correct"]}) for i in range(0, len(rows), n)
            ]))
    return result


def bootstrap(first: list, second: list, replicates: int) -> dict:
    """Independently resample aligned differences with the registered RNG and quantiles."""
    a, b = ({row["problem_id"]: int(row["correct"]) for row in rows} for rows in (first, second))
    if a.keys() != b.keys() or len(a) != len(first) or len(b) != len(second):
        raise ValueError("Unaligned or repeated bootstrap identities")
    delta = np.array([a[key] - b[key] for key in sorted(a)])
    rng = np.random.default_rng(20261107)
    samples = np.concatenate([delta[rng.integers(len(delta), size=(min(256, replicates - i), len(delta)))].mean(1)
                              for i in range(0, replicates, 256)])
    low, high = np.quantile(samples, [0.025, 0.975])
    return dict(difference=float(delta.mean()), low=float(low), high=float(high), n=len(delta))


def export_description(directory: Path) -> dict | None:
    """Inspect config and tensor headers, without constructing or running a model.

    Safetensors reader: https://huggingface.co/docs/safetensors/index .
    Counts are stored unique tensor elements, not activation or optimizer memory.
    """
    from safetensors import safe_open

    files = sorted(directory.glob("*.safetensors"))
    if not files or not (directory / "config.json").exists():
        return None
    shapes = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in shapes:
                    raise ValueError("Duplicate exported tensor")
                shapes[name] = handle.get_slice(name).get_shape()
    return {"architecture": read(directory / "config.json"), "unique_tensors": len(shapes),
            "parameter_elements": sum(math.prod(shape) for shape in shapes.values())}


def verify_export(directory: Path, experiment: dict, condition: str) -> dict:
    """Compare each BF16 export tensor with the saved FP32 master; never execute it."""
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    description = export_description(directory / "model")
    if description is None:
        raise ValueError(f"Restore the separately published model export: {directory / 'model'}")
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu", mmap=True)
    master = checkpoint["master"]
    if checkpoint["generation"] != experiment["conditions"][condition]["generation"]:
        raise ValueError("Checkpoint generation mismatch")
    seen = set()
    for path in sorted((directory / "model").glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                value = handle.get_tensor(name)
                if name not in master or not torch.equal(value, master[name].to(value.dtype)):
                    raise ValueError(f"Export differs from master: {name}")
                seen.add(name)
    if seen != master.keys() or any(t.dtype != torch.float32 for t in master.values()):
        raise ValueError("Master tensor coverage or precision mismatch")
    return {**description, "generation": checkpoint["generation"], "export_matches_master_cast": True}


def verify_trace(directory: Path, data: dict, config: dict) -> dict:
    """Regrade logged responses and reconcile committed candidate metrics and weights.

    Repeated responses from restart/retry count as resource use; they are deduplicated
    only for committed-generation reconciliation and must agree in content.
    """
    lookup = {row["id"]: row for rows in data.values() for row in rows}
    schedules, seen, candidates = {}, {}, defaultdict(list)
    counts, duplicates = defaultdict(lambda: Counter()), 0
    for line in (directory / "events.jsonl").open(encoding="utf-8"):
        event = json.loads(line)
        if event.get("event") == "candidate" and event.get("phase") == "train":
            seeds = random.Random(f"es-v1:{config['seed']}:{event['generation']}").sample(range(2**62), config["population"])
            if event["seed"] != seeds[event["candidate"]]:
                raise ValueError("Logged perturbation seed differs from paired schedule")
        if event.get("event") != "response":
            continue
        check_row(event, lookup[event["problem_id"]])
        phase, condition = event["phase"], event.get("condition", "calibration")
        counts[f"{condition}/{phase}"].update(responses=1, generated_tokens=event["generated_tokens"])
        if phase != "train":
            continue
        generation = event["generation"]
        if generation not in schedules:
            schedules[generation] = [row["id"] for row in scheduled_rows(data, config, generation)]
        if event["problem_id"] not in schedules[generation]:
            raise ValueError("Training prompt schedule differs (possible split leakage)")
        key = condition, generation, event["candidate"], event["problem_id"]
        digest = hashlib.sha256(json.dumps({k: event[k] for k in (
            "response", "token_ids", "correct", "valid", "reward", "structure", "truncated")}, sort_keys=True).encode()).hexdigest()
        if key in seen:
            if seen[key] != digest:
                raise ValueError("Replayed training response differs")
            duplicates += 1
            continue
        seen[key] = digest
        position = len(candidates[key[:3]])
        if event["problem_id"] != schedules[generation][position] or event["offset"] != position // config["batch_size"] * config["batch_size"]:
            raise ValueError("Training prompt order or batch offset differs")
        candidates[key[:3]].append({k: event[k] for k in ("reward", "correct", "valid", "generated_tokens", "truncated")})
    diagnostics = {}
    first = config.get("continuation", {}).get("parent_generation", 0)
    for condition in ("main", "reward_only"):
        metrics = read(directory / condition / "metrics.json")
        changed, positive, largest = 0, 0, 0.0
        for metric in metrics:
            q = np.asarray(metric["candidate_quality"], dtype=np.float64)
            novelty = np.asarray(metric["candidate_novelty"], dtype=np.float32)
            base = ((q - q.mean()) / (q.std() + 1e-8)).astype(np.float32)
            weight = base + np.float32(config["novelty_weight"] if condition == "main" else 0) * novelty
            if not np.allclose(weight, metric["candidate_weights"], rtol=0, atol=2e-7):
                raise ValueError("Recorded update coefficients differ from amended equation")
            changed += int(np.count_nonzero(weight != base))
            positive += int(np.count_nonzero(novelty > 0))
            largest = max(largest, float(np.max(np.abs(weight - base))))
            generation = metric["generation"] - 1
            if generation < first:
                continue  # Parent prefix is checked against the parent run separately.
            rows = []
            for index in range(config["population"]):
                candidate = candidates[condition, generation, index]
                if len(candidate) != config["rotating_count"] + config["probe_count"]:
                    raise ValueError("Incomplete committed candidate trace")
                if not math.isclose(np.mean([r["reward"] for r in candidate]), q[index], abs_tol=1e-14):
                    raise ValueError("Candidate quality differs from raw responses")
                rows.extend(candidate)
            for key in ("generated_tokens", "truncated"):
                if sum(row[key] for row in rows) != metric[key]:
                    raise ValueError(f"Committed generation {key} differs")
            for key, raw in (("accuracy", "correct"), ("validity", "valid"), ("quality", "reward")):
                if not math.isclose(float(np.mean([row[raw] for row in rows])), metric[key], abs_tol=1e-14):
                    raise ValueError(f"Committed generation {key} differs")
        diagnostics[condition] = dict(generations=len(metrics), candidates_weight_changed=changed,
                                      candidates_positive_novelty=positive, largest_absolute_weight_change=largest)
    return dict(response_events=dict(counts), repeated_training_responses=duplicates, coefficients=diagnostics)


def verify_all(pilot: Path, continuation: Path, data_directory: Path, fresh_directory: Path, output: Path) -> dict:
    """Hash inputs before/after and save a new verification revision outside every input."""
    inputs = {"pilot": pilot, "continuation": continuation, "data": data_directory, "fresh": fresh_directory}
    separate_output(output, list(inputs.values()))
    print("Hashing source evidence (excluding live dashboard logs, awake.log, writer.lock and download cache)", flush=True)
    before = {name: inventory(path) for name, path in inputs.items()}
    data, hashes = datasets([data_directory, fresh_directory])
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset_hashes": hashes,
              "input_files": before, "studies": {}, "scope": "Existing records only; no model execution or training."}
    result["inventory_exclusions"] = ["dashboard-error.log", "dashboard.log", "awake.log", "writer.lock", ".cache/"]
    for name, directory in (("pilot", pilot), ("continuation", continuation)):
        print(f"Verifying existing {name} records", flush=True)
        experiment = read(directory / "experiment.json")
        config = experiment["config"]
        available = data if name == "continuation" else {k: v for k, v in data.items() if k != "fresh"}
        if experiment["dataset_manifest"]["splits"] != {k: hashes[k] for k in available}:
            raise ValueError("Run manifest differs from explicit input datasets")
        expected = scheduled_rows(available, config)
        evaluations, raw = {}, {}
        for condition in ("original", "main", "reward_only", "parent_main", "parent_reward_only"):
            path = directory / condition / "evaluation.json"
            if condition.startswith("parent_") and name == "pilot":
                continue
            raw[condition] = read(path)
            evaluations[condition] = verify_evaluation(raw[condition],
                {"fresh": expected["fresh"]} if condition.startswith("parent_") else expected, config)
        split = config.get("primary_split", "test")
        result["studies"][name] = {
            "evaluations": evaluations,
            "primary": bootstrap(raw["main"]["sets"][split], raw["reward_only"]["sets"][split], config["bootstrap_replicates"]),
            "trace": verify_trace(directory, available, config),
            "exports": {c: verify_export(directory / c, experiment, c) for c in ("main", "reward_only")},
            "phase_boundary_overrun_seconds": {
                phase: max(0, value.get("finished_at", value["deadline"]) - value["deadline"])
                for phase, value in experiment["phases"].items()
            },
        }
    for condition in ("main", "reward_only"):
        parent = read(pilot / condition / "metrics.json")
        if read(continuation / condition / "metrics.json")[:len(parent)] != parent:
            raise ValueError("Continuation metric prefix differs from parent")
    original = read(pilot / "original/evaluation.json")["sets"]
    reused = read(continuation / "original/evaluation.json")["sets"]
    if any(reused[split] != rows for split, rows in original.items()):
        raise ValueError("Reused original-model outcomes differ from the pinned pilot")
    result["reused_original_evaluations_match"] = True
    print("Rechecking source hashes after verification", flush=True)
    after = {name: inventory(path) for name, path in inputs.items()}
    if before != after:
        output.mkdir(parents=True, exist_ok=True)
        (output / "changed-inputs.json").write_text(json.dumps({
            name: [key for key in before[name].keys() | after[name].keys()
                   if before[name].get(key) != after[name].get(key)] for name in inputs
        }, indent=2), encoding="utf-8")
        raise ValueError("Source evidence changed during verification")
    result["source_files_unchanged"] = True
    result["evidence_gaps"] = [
        "Generation-zero tensors were not retained; old initialization-audit assertions are historical evidence only.",
        "No complete per-generation embedding/archive trajectory is retained; distances cannot all be recomputed without model inference.",
        "Wall intervals are not GPU compute time; the unlogged outage tail is unknown.",
        "One training seed cannot establish seed robustness; split separation cannot rule out pretrained-data contamination.",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
