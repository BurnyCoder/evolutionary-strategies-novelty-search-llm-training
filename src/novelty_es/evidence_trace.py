"""Reconcile prompt, response and evaluation events against the frozen schedules.

Tokenization uses the exported vocabulary and chat template, without network access
or neural execution. https://huggingface.co/docs/tokenizers/api/tokenizer .
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from jinja2 import Environment
from tokenizers import Tokenizer

from .scientific_audit import grade_registered, mismatch
from .verification import read, scheduled_rows


def fingerprint(row):
    """Include the full observed response identity; batch time may differ on retries."""
    keys = (
        "problem_id",
        "prompt",
        "numbers",
        "target",
        "token_ids",
        "response",
        "reward",
        "valid",
        "correct",
        "structure",
        "expression",
        "generated_tokens",
        "truncated",
    )
    return hashlib.sha256(json.dumps([row[k] for k in keys], separators=(",", ":")).encode()).digest()


def evaluation_batches(raw, config, continued):
    """Identify expected committed batches and exclude only explicitly reused pilot records."""
    result = {}
    for arm, evaluation in raw.items():
        for split, rows in evaluation["sets"].items():
            if continued and arm == "original" and split != "fresh":
                continue
            for start in range(0, len(rows), config["batch_size"]):
                result[arm, split, start] = [fingerprint(r) for r in rows[start : start + config["batch_size"]]]
    return result


def population_changes(metrics, changes, config):
    """Analyze the same observed population. Preserve raw novelty when only task reward changes."""
    result = []
    count = config["probe_count"] + config["rotating_count"]
    for (arm, gen), candidates in sorted(changes.items()):
        row = metrics[arm][gen]
        original = np.array(row["candidate_quality"], dtype=np.float64)
        strict = original.copy()
        for index, differences in candidates.items():
            strict[index] += sum(x["registered_reward"] - x["historical_reward"] for x in differences) / count

        def standardize(x):
            return (np.zeros_like(x) if (x == x[0]).all() else (x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)

        bonus = np.asarray(row["candidate_novelty"], dtype=np.float32) * (
            config["novelty_weight"] if arm == "main" else 0
        )
        old_w, new_w = standardize(original) + bonus, standardize(strict) + bonus
        if not np.allclose(old_w, row["candidate_weights"], atol=2e-7, rtol=0):
            raise ValueError("Historical coefficients differ")
        for index, differences in candidates.items():
            for item in differences:
                item.update(
                    historical_candidate_quality=float(original[index]),
                    registered_candidate_quality=float(strict[index]),
                    recorded_weight=float(row["candidate_weights"][index]),
                    fixed_novelty_registered_weight=float(new_w[index]),
                )
        result.append(
            dict(
                condition=arm,
                update=gen + 1,
                changed_candidates=len(candidates),
                changed_weights_holding_recorded_novelty=int(np.count_nonzero(old_w != new_w)),
                max_abs_weight_change=float(np.max(np.abs(old_w - new_w))),
            )
        )
    return result


def inspect_trace(directory, data, config, raw):
    """Stream traces once. Uncommitted and repeated attempts stay visible in the accounting."""
    continued = "continuation" in config
    expected_eval = evaluation_batches(raw, config, continued)
    metrics = {arm: read(directory / arm / "metrics.json") for arm in ("main", "reward_only")}
    first_divergence = next(
        (
            i
            for i, (a, b) in enumerate(zip(metrics["main"], metrics["reward_only"], strict=True))
            if a["candidate_weights"] != b["candidate_weights"]
        ),
        len(metrics["main"]) - 1,
    )
    lookup = {r["id"]: r for split in data.values() for r in split}
    mdir = directory / "main/model"
    tokenizer = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(
        (mdir / "chat_template.jinja").read_text(encoding="utf-8")
    )
    decoding = read(mdir / "generation_config.json")
    eos = set(decoding["eos_token_id"])

    @lru_cache(maxsize=16384)
    def rendered(prompt):
        return template.render(messages=[dict(role="user", content=prompt)], add_generation_prompt=True)

    @lru_cache(maxsize=8192)
    def decoded(tokens):
        return tokenizer.decode(list(tokens), skip_special_tokens=True)

    committed, attempts, counts = Counter(), Counter(), Counter()
    seen, prefix, times, val = {}, {}, {}, defaultdict(dict)
    deltas, details, other, launches = defaultdict(lambda: defaultdict(list)), [], [], []
    train_schedule = {}
    active, prompts, outputs = None, [], []

    def finish_batch():
        if active is None:
            return
        phase, arm, _, _, split, offset = active
        if len(outputs) < len(prompts):
            counts["prompted_without_final_response"] += len(prompts) - len(outputs)
        if phase == "evaluation":
            key = arm, split, offset
            if key not in expected_eval:
                raise ValueError(f"Unscheduled evaluation batch {key}")
            attempts[key] += 1
            if outputs == expected_eval[key]:
                committed[key] += 1
            else:
                counts["uncommitted_or_incomplete_evaluation_attempts"] += 1

    with (directory / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            e = json.loads(line)
            kind = e["event"]
            if kind == "pilot_entered":
                launches.append({k: e[k] for k in ("timestamp", "phase", "code_revision")})
                finish_batch()
                active, prompts, outputs = None, [], []
            if kind not in {"prompt", "response"}:
                continue
            phase, arm, gen, cand, split, offset = key = tuple(
                e.get(k) for k in ("phase", "condition", "generation", "candidate", "split", "offset")
            )
            pid = e["problem_id"]
            if kind == "prompt":
                if key != active or outputs:
                    finish_batch()
                    active, prompts, outputs = key, [], []
                problem = lookup[pid]
                expected_prompt = (
                    f"Use the numbers {problem['numbers']} exactly once each to make {problem['target']}. "
                    "You may use +, -, *, / and parentheses. Do not use other numbers or operations. "
                    "Return only <answer>expression</answer>."
                )
                if e["prompt"] != expected_prompt or e["rendered_prompt"] != rendered(expected_prompt):
                    raise ValueError(f"Rendered prompt mismatch {key}/{pid}")
                if phase.startswith("calibration") or phase == "validation":
                    permitted = "validation" if "validation" in phase else "train"
                    if not pid.startswith(permitted + "-"):
                        raise ValueError("Held-out exposure during calibration or validation")
                prompts.append(pid)
                counts["rendered_prompts"] += 1
                timing = times.setdefault("/".join(filter(None, (phase, arm, split))), dict(first=e["timestamp"]))
                timing["last"] = e["timestamp"]
                continue
            if active != key or len(outputs) >= len(prompts) or prompts[len(outputs)] != pid:
                raise ValueError(f"Response and prompt identities differ {key}/{pid}")
            tokens = tuple(e["token_ids"])
            if (
                not tokens
                or len(tokens) != e["generated_tokens"]
                or decoded(tokens) != e["response"]
                or any(t in eos for t in tokens[:-1])
                or len(tokens) > config["max_new_tokens"]
                or e["truncated"] != (len(tokens) == config["max_new_tokens"] and tokens[-1] not in eos)
            ):
                raise ValueError(f"Token decoding or termination mismatch {key}/{pid}")
            digest = fingerprint(e)
            outputs.append(digest)
            counts["decoded_responses"] += 1
            target = lookup[pid]
            reward, correct, valid, structure = grade_registered(e["response"], target["numbers"], target["target"])
            differs = (reward, correct, valid, structure) != (e["reward"], e["correct"], e["valid"], e["structure"])
            if phase == "validation":
                if pid != data["validation"][offset + len(outputs) - 1]["id"]:
                    raise ValueError("Wrong validation subset or order")
                val[arm, gen][pid] = e["correct"], e["valid"], e["reward"], len(tokens), e["truncated"]
            if phase != "train":
                if differs:
                    other.append(
                        dict(
                            phase=phase,
                            condition=arm,
                            split=split,
                            offset=offset,
                            timestamp=e["timestamp"],
                            **mismatch(e, reward, correct),
                        )
                    )
                continue
            if gen not in train_schedule:
                train_schedule[gen] = [r["id"] for r in scheduled_rows(data, config, gen)]
            pos = offset + len(outputs) - 1
            if train_schedule[gen][pos] != pid:
                raise ValueError(f"Training order mismatch {key}/{pid}")
            identity = arm, gen, cand, pid
            if identity in seen:
                if seen[identity] != digest:
                    raise ValueError("Conflicting repeated training response")
                counts["repeated_training_responses"] += 1
                continue
            seen[identity] = digest
            if not continued and gen <= first_divergence:
                prefix[gen, cand, pid, arm] = digest
            if differs:
                item = dict(
                    condition=arm,
                    generation_index=gen,
                    update=gen + 1,
                    candidate=cand,
                    timestamp=e["timestamp"],
                    fixed_probe=pos >= config["rotating_count"],
                    **mismatch(e, reward, correct),
                )
                item["novelty_eligibility_changed"] = item["fixed_probe"] and correct != e["correct"]
                details.append(item)
                deltas[arm, gen][cand].append(item)
    finish_batch()
    if expected_eval.keys() != committed.keys():
        raise ValueError(f"Evaluation batch lacks matching trace: {list(expected_eval.keys() - committed.keys())[:5]}")
    start = config.get("continuation", {}).get("parent_generation", 0)
    if any(len(metrics[a]) != len(metrics["main"]) for a in metrics):
        raise ValueError("Unmatched generation counts")
    expected_n = (
        2 * (len(metrics["main"]) - start) * config["population"] * (config["probe_count"] + config["rotating_count"])
    )
    if len(seen) != expected_n:
        raise ValueError("Missing or unexpected committed training rows")
    registered_val = {(a, r["generation"]) for a, items in metrics.items() for r in items[start:] if "validation" in r}
    if val.keys() != registered_val:
        raise ValueError("Validation checkpoint coverage differs")
    for (arm, gen), rows in val.items():
        fields = list(zip(*rows.values(), strict=True))
        observed = metrics[arm][gen - 1]["validation"]
        if (
            len(rows) != config["validation_count"]
            or any(
                not math.isclose(sum(v) / len(v), observed[k])
                for v, k in zip(fields[:3], ("accuracy", "validity", "quality"), strict=True)
            )
            or sum(fields[3]) != observed["generated_tokens"]
            or sum(fields[4]) != observed["truncated"]
        ):
            raise ValueError("Validation metric differs from saved responses")
    pairs = [(value, prefix[g, c, p, "reward_only"]) for (g, c, p, a), value in prefix.items() if a == "main"]
    return dict(
        counts=dict(counts),
        unique_training_rows=len(seen),
        committed_evaluation_batches=len(committed),
        evaluation_attempts=sum(attempts.values()),
        additional_identical_evaluation_batches=sum(committed.values()) - len(committed),
        validation_populations=len(val),
        phase_prompt_times=times,
        execution_revisions=launches,
        first_different_coefficient_update=first_divergence + 1,
        pre_update_paired_responses=len(pairs),
        pre_update_different_responses=sum(x != y for x, y in pairs),
        decoding=decoding,
        training_discrepancies=details,
        affected_populations=population_changes(metrics, deltas, config),
        other_event_discrepancies=other,
    )
