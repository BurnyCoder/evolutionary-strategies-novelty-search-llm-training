"""Solvable Countdown data and an exact, non-executing arithmetic verifier.

The verifier uses Python AST parsing, never eval:
https://docs.python.org/3.12/library/ast.html
Exact division uses https://docs.python.org/3.12/library/fractions.html .
Generation is our synthetic protocol, not the Countdown distribution of ES at Scale.
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class Grade:
    """Keep correctness separate from the positive legality-only reward."""

    reward: float = 0.0
    correct: bool = False
    valid: bool = False
    expression: str = ""
    structure: str = ""


def _evaluate(node: ast.AST) -> tuple[Fraction, str]:
    """Recursively compute an allowed expression and its canonical syntax tree."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value), str(node.value)
    if not isinstance(node, ast.BinOp):
        raise ValueError("Only integer literals and binary arithmetic are allowed")
    left, left_tree = _evaluate(node.left)
    right, right_tree = _evaluate(node.right)
    if isinstance(node.op, ast.Add):
        value, symbol = left + right, "+"
    elif isinstance(node.op, ast.Sub):
        value, symbol = left - right, "-"
    elif isinstance(node.op, ast.Mult):
        value, symbol = left * right, "*"
    elif isinstance(node.op, ast.Div):
        value, symbol = left / right, "/"
    else:
        raise ValueError("Disallowed operator")
    if symbol in {"+", "*"}:
        left_tree, right_tree = sorted((left_tree, right_tree))
    return value, f"({symbol} {left_tree} {right_tree})"


def grade(response: str, numbers: list[int], target: int) -> Grade:
    """Grade one answer; permit one answer tag or a bare expression, never code.

    Commutative child order is canonicalized, but associativity/distributivity are
    not collapsed: structure diversity measures syntax, not semantic equivalence.
    """
    if len(response) > 8192:
        return Grade()
    answers = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if len(answers) > 1:
        return Grade()
    expression = (answers[0] if answers else response).strip()
    if not re.fullmatch(r"[0-9+*/()\s-]{1,512}", expression):
        return Grade()
    try:
        tree = ast.parse(expression, mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > 64:
            return Grade()
        literals = [n.value for n in nodes if isinstance(n, ast.Constant)]
        if any(type(n) is not int for n in literals) or Counter(literals) != Counter(numbers):
            return Grade()
        value, structure = _evaluate(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return Grade()
    correct = value == target
    return Grade(1.0 if correct else 0.1, correct, True, expression, structure)


def make_prompt(row: dict) -> str:
    """Expose only numbers and target; the solvability witness stays private to grading."""
    return (
        f"Use the numbers {row['numbers']} exactly once each to make {row['target']}. "
        "You may use +, -, *, / and parentheses. Do not use other numbers or operations. "
        "Return only <answer>expression</answer>."
    )


def _witness(numbers: list[int], rng: random.Random) -> tuple[int, str] | None:
    """Sample an arithmetic tree and retain a bounded positive integer root."""
    terms = [(Fraction(n), str(n)) for n in numbers]
    while len(terms) > 1:
        left = terms.pop(rng.randrange(len(terms)))
        right = terms.pop(rng.randrange(len(terms)))
        symbol = rng.choice("+-*/")
        if symbol == "/" and right[0] == 0:
            return None
        value = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}[symbol](
            left[0], right[0]
        )
        terms.append((value, f"({left[1]}{symbol}{right[1]})"))
    value, expression = terms[0]
    if value.denominator == 1 and 1 <= value <= 999:
        return int(value), expression
    return None


def generate_dataset(
    seed: int, sizes: dict[str, int] | None = None, excluded_groups: set[tuple[int, ...]] | None = None
) -> dict[str, list[dict]]:
    """Generate unique multisets globally, making leakage by permutation impossible.

    Proposed numbers are independent uniform integers 1..50. Rejection of duplicate
    multisets and unsuitable sampled trees changes the accepted distribution;
    neither accepted problems nor their targets are uniformly sampled.
    """
    sizes = sizes or {"train": 8192, "validation": 512, "test": 2048, "hard": 1024}
    rng, seen, result = random.Random(seed), set(excluded_groups or ()), {}
    for split, count in sizes.items():
        rows = []
        while len(rows) < count:
            n = 5 if split == "hard" else 3 + len(rows) % 2
            numbers = [rng.randint(1, 50) for _ in range(n)]
            group = tuple(sorted(numbers))
            if group in seen:
                continue
            solution = _witness(numbers, rng)
            if solution is None:
                continue
            target, witness = solution
            seen.add(group)
            rows.append({"id": f"{split}-{len(rows):05d}", "numbers": numbers, "target": target, "witness": witness})
        result[split] = rows
    return result


def prepare_data(
    directory: Path, seed: int, sizes: dict[str, int] | None = None,
    excluded_groups: set[tuple[int, ...]] | None = None,
) -> dict:
    """Write deterministic JSONL and a content-hash manifest before preregistration."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "generator": "sampled-tree-v1", "splits": {}}
    for split, rows in generate_dataset(seed, sizes, excluded_groups).items():
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        (directory / f"{split}.jsonl").write_bytes(payload)
        manifest["splits"][split] = {"count": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_data(directory: Path) -> dict[str, list[dict]]:
    """Reject a changed dataset rather than silently continuing a registered run."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for split, info in manifest["splits"].items():
        payload = (directory / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {split}")
        result[split] = [json.loads(line) for line in payload.splitlines()]
        if len(result[split]) != info["count"]:
            raise ValueError(f"Dataset row count mismatch: {split}")
    return result
