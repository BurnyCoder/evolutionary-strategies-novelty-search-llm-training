"""Analytic checks for the registered pass@k and paired problem bootstrap."""

import numpy as np
import pytest

from novelty_es.analysis import paired_bootstrap, pass_at_k


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
