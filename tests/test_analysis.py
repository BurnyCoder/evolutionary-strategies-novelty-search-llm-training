"""Analytic checks for the registered pass@k and paired problem bootstrap."""

import numpy as np
import pytest

from novelty_es.analysis import novelty_intervention, paired_bootstrap, pass_at_k


def test_pass_estimator_edge_cases_and_reference():
    assert pass_at_k(16, 0, 8) == 0
    assert pass_at_k(16, 16, 8) == 1
    assert pass_at_k(16, 1, 8) == 0.5
    assert pass_at_k(16, 4, 1) == 0.25
    with pytest.raises(ValueError):
        pass_at_k(7, 3, 8)


def test_paired_bootstrap_uses_paired_rows():
    identical = np.array([0, 1, 0, 1, 1])
    result = paired_bootstrap(identical, identical, replicates=100)
    assert result == {"difference": 0.0, "low": 0.0, "high": 0.0, "n": 5}
    result = paired_bootstrap(np.ones(5), np.zeros(5), replicates=100)
    assert result["low"] == result["high"] == result["difference"] == 1.0


def test_novelty_activation_is_distinct_from_an_optimizer_intervention():
    """A positive common bonus need not change ES; breaking a quality tie does."""
    rows = [
        {"candidate_quality": [0.0, 0.1], "candidate_novelty": [0.0, 0.0]},
        {"candidate_quality": [0.0, 0.1], "candidate_novelty": [0.5, 0.5]},
        {"candidate_quality": [0.1, 0.1], "candidate_novelty": [0.0, 1.0]},
    ]
    result = novelty_intervention(rows, 0.25)
    assert result["generations_observed"] == 3
    assert result["candidates_observed"] == 6
    assert result["candidates_positive_novelty"] == 3
    assert result["generations_positive_novelty"] == 2
    assert result["generations_rank_changed"] == 1
    assert result["candidates_rank_changed"] == 2
    assert result["largest_absolute_rank_change"] == 0.5


def test_novelty_diagnostic_replays_float32_fitness_and_handles_missing_runs():
    """A bonus rounded away in training must not count as an effective rank change."""
    rows = [{"candidate_quality": [1.0, 1.0], "candidate_novelty": [0.0, 1e-12]}]
    assert novelty_intervention(rows, 0.25)["generations_rank_changed"] == 0
    assert novelty_intervention([], 0.25)["generations_observed"] == 0
