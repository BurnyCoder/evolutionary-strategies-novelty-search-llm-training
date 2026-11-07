"""Registered estimators, independent of model inference and report formatting.

pass@k: Chen et al. https://arxiv.org/abs/2107.03374 .
Resampling uses NumPy's explicit Generator API, https://numpy.org/doc/stable/reference/random/generator.html .
Intervals are conditional on the realized training seed, not seed-robust inference.
"""

import math

import numpy as np
import torch

from .optimizer import candidate_weights


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Probability that a uniformly drawn k-subset contains a correct sampled answer."""
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= correct <= n and 1 <= k <= n")
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def paired_bootstrap(main, control, replicates: int = 10000, seed: int = 20261107) -> dict:
    """Resample aligned problems, retaining cross-condition dependence in each draw."""
    difference = np.asarray(main, dtype=float) - np.asarray(control, dtype=float)
    if difference.ndim != 1 or not len(difference) or len(main) != len(control):
        raise ValueError("Nonempty, aligned problem arrays required")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 256):
        indices = rng.integers(len(difference), size=(min(256, replicates - start), len(difference)))
        means.extend(difference[indices].mean(axis=1).tolist())
    low, high = np.quantile(means, [0.025, 0.975])
    return {"difference": float(difference.mean()), "low": float(low), "high": float(high), "n": len(difference)}


def novelty_intervention(generations: list[dict], coefficient: float) -> dict:
    """Check bonus activation and weight changes on each recorded candidate population.

    This is a descriptive, post-hoc local counterfactual, not a replay of a whole
    training trajectory. Reuse the optimizer's coefficient function so positive
    bonuses rounded away in FP32 do not count as changes.
    """
    result = {
        "generations_observed": len(generations),
        "candidates_observed": 0,
        "candidates_positive_novelty": 0,
        "generations_positive_novelty": 0,
        "generations_weight_changed": 0,
        "candidates_weight_changed": 0,
        "largest_absolute_weight_change": 0.0,
    }
    for row in generations:
        quality = torch.tensor(row["candidate_quality"], dtype=torch.float64)
        novelty = torch.tensor(row["candidate_novelty"], dtype=torch.float32)
        if quality.shape != novelty.shape or quality.ndim != 1 or not len(quality):
            raise ValueError("Aligned nonempty candidate quality and novelty vectors required")
        difference = candidate_weights(quality, novelty, coefficient) - candidate_weights(quality, novelty, 0.0)
        positive, changed = int((novelty > 0).sum()), int((difference != 0).sum())
        result["candidates_observed"] += len(quality)
        result["candidates_positive_novelty"] += positive
        result["generations_positive_novelty"] += bool(positive)
        result["generations_weight_changed"] += bool(changed)
        result["candidates_weight_changed"] += changed
        result["largest_absolute_weight_change"] = max(
            result["largest_absolute_weight_change"], float(difference.abs().max())
        )
    return result
