"""Reanalysis of the frozen written scoring rule. No training or inference imports.

The experiment's historical scorer is deliberately retained for reproducibility.
See reports/preregistration.md, Chen et al. https://arxiv.org/abs/2107.03374
and Fay & Lumbard https://pmc.ncbi.nlm.nih.gov/articles/PMC9447366/ .
"""

import ast
import math
import operator
import re
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

INVALID = (0.0, False, False, "")
OPERATIONS = {
    ast.Add: ("+", operator.add),
    ast.Sub: ("-", operator.sub),
    ast.Mult: ("*", operator.mul),
    ast.Div: ("/", operator.truediv),
}


def grade_registered(text, numbers, target):
    """Enforce one balanced literal tag pair or a bare expression, and exact rational arithmetic."""
    if len(text) > 8192:
        return INVALID
    opens, closes = text.count("<answer>"), text.count("</answer>")
    if opens or closes:
        if (opens, closes) != (1, 1):
            return INVALID
        prefix, _, rest = text.partition("<answer>")
        text, end, _ = rest.partition("</answer>")
        if not end or "</answer>" in prefix:
            return INVALID
    expr = text.strip()
    if not re.fullmatch(r"[0-9+*/()\s-]{1,512}", expr):
        return INVALID
    leaves = []

    def visit(node):
        if type(node) is ast.Constant and type(node.value) is int:
            leaves.append(node.value)
            return Fraction(node.value), str(node.value)
        if type(node) is not ast.BinOp or type(node.op) not in OPERATIONS:
            raise ValueError("Outside registered arithmetic")
        symbol, operation = OPERATIONS[type(node.op)]
        left, a = visit(node.left)
        right, b = visit(node.right)
        if symbol in {"+", "*"}:
            a, b = sorted((a, b))
        return operation(left, right), f"({symbol} {a} {b})"

    try:
        tree = ast.parse(expr, mode="eval")
        if len(list(ast.walk(tree))) > 64:
            return INVALID
        value, structure = visit(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return INVALID
    if Counter(leaves) != Counter(numbers):
        return INVALID
    correct = value == target
    return (1.0 if correct else 0.1, correct, True, structure)


def exact_paired_p(main_only, control_only):
    """Two-sided exact conditional McNemar test, including the no-discordances case."""
    discordant = main_only + control_only
    tail = sum(math.comb(discordant, i) for i in range(min(main_only, control_only) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**discordant)))


def interval(values, replicates):
    """Registered paired percentile interval in units of probability, including changes in changes."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("A nonempty vector of problem differences is required")
    rng = np.random.default_rng(20261107)
    means = [
        values[rng.integers(len(values), size=(min(100, replicates - i), len(values)))].mean(axis=1)
        for i in range(0, replicates, 100)
    ]
    low, high = np.percentile(np.concatenate(means), [2.5, 97.5])
    return dict(n=len(values), difference=float(values.mean()), low=float(low), high=float(high))


def paired_result(main, control, replicates):
    """Match IDs first. The exact test is a post-hoc check, separate from the registered interval."""
    if main.keys() != control.keys():
        raise ValueError("Unmatched problem IDs")
    delta = [int(main[k]) - int(control[k]) for k in sorted(main)]
    a, b = delta.count(1), delta.count(-1)
    return dict(
        **interval(delta, replicates),
        main_only=a,
        control_only=b,
        both_correct=sum(bool(main[k] and control[k]) for k in main),
        both_wrong=sum(not (main[k] or control[k]) for k in main),
        exact_two_sided_p=exact_paired_p(a, b),
    )


def require_endpoint(condition, target):
    """A complete flag alone cannot establish the registered checkpoint."""
    if not condition.get("complete") or (condition.get("generation"), condition.get("generations")) != (target, target):
        raise ValueError("Incomplete or wrong scheduled checkpoint")


def sampled_statistics(groups, n):
    """Equal weight per problem, with all scheduled sample indices present exactly once."""
    counts, structures = [], []
    for rows in groups.values():
        if set(rows) != set(range(n)):
            raise ValueError("Incomplete sampled problem")
        counts.append(sum(correct for correct, _ in rows.values()))
        structures.append(len({tree for correct, tree in rows.values() if correct}))
    return {
        **{f"pass{k}": float(np.mean([1 - math.comb(n - c, k) / math.comb(n, k) for c in counts])) for k in (1, 8)},
        "mean_unique_correct_structures": float(np.mean(structures)),
        "problems": len(groups),
    }


def evaluation_statistics(rows, expected, split, config, differences, condition):
    """Independently grade every committed row and preserve both readings in the evidence."""
    if len(rows) != len(expected):
        raise ValueError(f"Incomplete evaluation: {condition}/{split}")
    outcomes, groups, totals = {}, defaultdict(dict), Counter()
    for row, problem in zip(rows, expected, strict=True):
        key, sample = row["problem_id"], row.get("sample")
        if (key, sample) != (problem["id"], problem.get("sample")) or (row["numbers"], row["target"]) != (
            problem["numbers"],
            problem["target"],
        ):
            raise ValueError(f"Swapped or unexpected evaluation identity: {condition}/{split}")
        reward, correct, valid, structure = grade_registered(row["response"], problem["numbers"], problem["target"])
        if sample in groups[key]:
            raise ValueError("Duplicate response identity")
        groups[key][sample] = (correct, structure)
        if split != "sampled":
            outcomes[key] = correct
        if (reward, correct, valid, structure) != (row["reward"], row["correct"], row["valid"], row["structure"]):
            differences.append(dict(condition=condition, split=split, sample=sample, **mismatch(row, reward, correct)))
        totals.update(
            responses=1,
            correct=correct,
            valid=valid,
            historical_correct=row["correct"],
            historical_valid=row["valid"],
            generated_tokens=row["generated_tokens"],
            truncated=row["truncated"],
        )
    result = dict(totals)
    result["reward_sum"] = totals["correct"] + (totals["valid"] - totals["correct"]) / 10
    result["historical_reward_sum"] = (
        totals["historical_correct"] + (totals["historical_valid"] - totals["historical_correct"]) / 10
    )
    if split == "sampled":
        result.update(sampled_statistics(groups, config["samples_per_problem"]))
    else:
        result["accuracy_interval"] = interval(list(outcomes.values()), config["bootstrap_replicates"])
    return result, outcomes


def mismatch(row, reward, correct):
    """Store the locator and exact change. The raw response stays at the pinned source."""
    return dict(
        problem_id=row["problem_id"],
        opening_tags=row["response"].count("<answer>"),
        closing_tags=row["response"].count("</answer>"),
        historical_reward=row["reward"],
        registered_reward=reward,
        historical_correct=row["correct"],
        registered_correct=correct,
    )
