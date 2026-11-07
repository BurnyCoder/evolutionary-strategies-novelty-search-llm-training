"""Analytic and replay checks for full-parameter, seed-reconstructed ES."""

import torch
from torch import nn

from novelty_es.optimizer import FullWeightES, centered_ranks, perturbation_seeds


def test_ties_and_flat_fitness():
    assert torch.equal(centered_ranks([3, 3, 3]), torch.zeros(3))
    assert torch.allclose(centered_ranks([1, 1, 3]), torch.tensor([-0.25, -0.25, 0.5]))
    assert torch.equal(centered_ranks([4]), torch.zeros(1))


def test_update_matches_analytic_gaussian_sum():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.zero_()
    es = FullWeightES(model, sigma=0.01, alpha=0.2, noise_device="cpu")
    seeds, fitness = [11, 12, 13], [1, 0, 2]
    noises = [es.noise("weight", model.weight.shape, seed) for seed in seeds]
    expected = sum(w * noise for w, noise in zip(centered_ranks(fitness), noises, strict=True)) * (0.2 / 3)
    es.update(seeds, fitness)
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
