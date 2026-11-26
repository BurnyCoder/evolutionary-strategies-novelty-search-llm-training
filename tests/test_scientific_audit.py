"""Model-free counterexamples for the registered arithmetic and paired endpoints."""

import pytest

from novelty_es.scientific_audit import (
    evaluation_statistics,
    exact_paired_p,
    grade_registered,
    paired_result,
    require_endpoint,
    sampled_statistics,
)
from novelty_es.verification import reference_grade


def test_extra_closing_tag_changes_the_gate():
    response = "<answer>33 - (33 - 23)</answer> explanation </answer>"
    assert reference_grade(response, [33, 33, 23], 23)["correct"]
    assert grade_registered(response, [33, 33, 23], 23) == (0.0, False, False, "")


@pytest.mark.parametrize(
    ("response", "reward"),
    [
        ("text <answer>6/(3/2)</answer> text", 1),
        ("(6-3)*2", 0.1),
        ("<answer>6/(3/2)</answer><answer>", 0),
        ("6//(3/2)", 0),
        ("<answer>6/(3/2)</answer><answer>4</answer>", 0),
        ("6/(3-3)", 0),
    ],
)
def test_registered_arithmetic(response, reward):
    assert grade_registered(response, [6, 3, 2], 4)[0] == reward


def test_exact_paired_and_identity_checks():
    assert exact_paired_p(0, 0) == 1
    assert exact_paired_p(0, 3) == 0.25
    assert exact_paired_p(3, 3) == 1
    assert paired_result({"a": 1, "b": 0}, {"b": 0, "a": 0}, 100)["main_only"] == 1
    with pytest.raises(ValueError):
        paired_result({"a": True}, {"b": True}, 100)


def test_incomplete_checkpoint_cannot_supply_a_registered_endpoint():
    with pytest.raises(ValueError):
        require_endpoint({"generation": 2, "generations": 2, "complete": True}, 3)


def test_hand_counted_subsets_and_missing_sample():
    rows = {i: (i == 0, "tree") for i in range(16)}
    assert sampled_statistics({"one": rows}, 16) == dict(
        pass1=1 / 16, pass8=0.5, mean_unique_correct_structures=1, problems=1
    )
    del rows[8]
    with pytest.raises(ValueError, match="Incomplete"):
        sampled_statistics({"one": rows}, 16)


def test_committed_rows_follow_registered_problem_identity():
    expected = [dict(id="a", numbers=[1], target=1), dict(id="b", numbers=[1], target=1)]
    row = dict(
        problem_id="a",
        numbers=[1],
        target=1,
        response="1",
        reward=1,
        correct=True,
        valid=True,
        structure="1",
        truncated=False,
        generated_tokens=1,
    )
    with pytest.raises(ValueError, match="Swapped"):
        evaluation_statistics([row, row], expected, "test", {}, [], "main")
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation_statistics([row, row], [expected[0], expected[0]], "test", {}, [], "main")
