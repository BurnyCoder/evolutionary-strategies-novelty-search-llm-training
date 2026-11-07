"""Analytic and replay checks for full-parameter, seed-reconstructed ES."""

import math

import pytest
import torch
from torch import nn

from novelty_es.optimizer import FullWeightES, candidate_weights, perturbation_seeds, z_scores


def test_ties_and_flat_fitness():
    assert torch.equal(z_scores([3, 3, 3]), torch.zeros(3))
    assert torch.allclose(z_scores([1, 1, 3]), torch.tensor([-1 / math.sqrt(2), -1 / math.sqrt(2), math.sqrt(2)]))
    assert torch.equal(z_scores([4]), torch.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        z_scores([1, float("nan")])


def test_novelty_is_added_after_quality_normalization():
    """A shared bonus survives; it cannot change the quality normalization denominator."""
    quality = [0.0, 0.1]
    baseline = candidate_weights(quality, [0.5, 0.5], 0.0)
    main = candidate_weights(quality, [0.5, 0.5], 1.0)
    assert torch.allclose(baseline, torch.tensor([-1.0, 1.0]))
    assert torch.allclose(main - baseline, torch.full((2,), 0.5))
    assert torch.equal(candidate_weights([1, 1], [0.2, 0.8], 1.0), torch.tensor([0.2, 0.8]))
    assert torch.equal(candidate_weights([1, 1], [0, 0], 1.0), torch.zeros(2))


def test_update_matches_analytic_gaussian_sum():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.zero_()
    es = FullWeightES(model, sigma=0.01, alpha=0.2, noise_device="cpu")
    seeds, weights = [11, 12, 13], [1.0, 0.0, 2.0]
    noises = [es.noise("weight", model.weight.shape, seed) for seed in seeds]
    expected = (noises[0] + 2 * noises[2]) * (0.2 / 3)
    es.update(seeds, weights)
    assert torch.allclose(es.master["weight"], expected, atol=1e-8)
    es.update(seeds, [0.0, 0.0, 0.0])
    assert torch.allclose(es.master["weight"], expected, atol=1e-8)


def test_reconstruction_does_not_drift_and_ties_update_once():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 2, bias=False)
            self.b = nn.Linear(3, 2, bias=False)
            self.b.weight = self.a.weight

    model = Tied()
    es = FullWeightES(model, 0.001, 0.0005, noise_device="cpu")
    assert len(es.master) == 1
    initial = model.a.weight.detach().clone()
    es.reconstruct(3)
    first = model.a.weight.detach().clone()
    es.reconstruct(9)
    es.reconstruct(3)
    assert torch.equal(model.a.weight, first)
    es.reconstruct(None)
    assert torch.equal(model.a.weight, initial)
    assert model.a.weight is model.b.weight


def test_noise_schedule_is_independent_of_condition():
    assert perturbation_seeds(0, 2, 30) == perturbation_seeds(0, 2, 30)
    assert perturbation_seeds(0, 2, 30) != perturbation_seeds(0, 3, 30)
    assert len(set(perturbation_seeds(0, 2, 30))) == 30
