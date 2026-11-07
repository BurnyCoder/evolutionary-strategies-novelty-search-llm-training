"""Protect the symbolic reward and disjoint groups before running any model."""
from fractions import Fraction

import pytest

from novelty_es.data import generate_dataset, grade, make_prompt


@pytest.mark.parametrize("response,numbers,target,reward", [
    ("<answer>(3+4)*2</answer>", [3, 4, 2], 14, 1.0),
    ("(3+4)*2", [3, 4, 2], 15, 0.1),
    ("(1/3)*3", [1, 3, 3], 1, 1.0),
    ("2+3", [2, 3, 4], 5, 0.0),
    ("2+2", [2, 3], 4, 0.0),
    ("2**3", [2, 3], 8, 0.0),
    ("__import__('os').system('echo unsafe')", [2, 3], 5, 0.0),
    ("True+2", [1, 2], 3, 0.0),
    ("1/(2-2)", [1, 2, 2], 1, 0.0),
    ("2.0+3", [2, 3], 5, 0.0),
    ("<answer>2+3</answer><answer>3+2</answer>", [2, 3], 5, 0.0),
])
def test_exact_reward(response, numbers, target, reward):
    """Wrong-but-legal answers must never pass the separate correctness gate."""
    result = grade(response, numbers, target)
    assert result.reward == reward
    assert result.correct == (reward == 1.0)
    assert result.valid == (reward > 0)


def test_structure_canonicalizes_commutative_order():
    assert grade("2+3", [2, 3], 5).structure == grade("3+2", [2, 3], 5).structure
    assert grade("3-2", [2, 3], 1).structure != grade("2-3", [2, 3], -1).structure


def test_dataset_is_reproducible_solvable_and_group_disjoint():
    sizes = {"train": 100, "validation": 12, "test": 24, "hard": 16}
    first = generate_dataset(7, sizes)
    assert first == generate_dataset(7, sizes)
    groups = set()
    for split, rows in first.items():
        assert len(rows) == sizes[split]
        for row in rows:
            group = tuple(sorted(row["numbers"]))
            assert group not in groups
            groups.add(group)
            assert grade(row["witness"], row["numbers"], row["target"]).correct
            assert row["witness"] not in make_prompt(row)
            assert Fraction(row["target"]).denominator == 1
