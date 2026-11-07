"""Independent implementation of Gaussian ES with source-aligned update weights.

Math reference: https://arxiv.org/abs/2509.24372 .
The upstream Academic Public License permits this project's noncommercial research
scope; source reuse must preserve its terms. This pilot uses an independent implementation.
Post-normalization novelty: https://arxiv.org/html/2606.24994v1, Equation 2.
PyTorch seeded generators: https://docs.pytorch.org/docs/stable/generated/torch.Generator.html .
FP32 masters live on the CPU; layer-sized GPU temporaries bound device memory.
"""

from __future__ import annotations

import hashlib
import random

import torch


def z_scores(values: list[float] | torch.Tensor) -> torch.Tensor:
    """ES at Scale population z-scores; equal quality contributes exactly zero.

    Algorithm 2 supplies the equation. The pinned reward_shaping.py uses a 1e-8
    denominator stabilizer and its trainer uses NumPy's population standard deviation.
    This implements the mathematical rule independently; it does not copy that source.
    """
    scores = torch.as_tensor(values, dtype=torch.float64)
    if scores.ndim != 1 or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("Quality must be a nonempty finite vector")
    if (scores == scores[0]).all():
        return torch.zeros(len(scores))
    return ((scores - scores.mean()) / (scores.std(correction=0) + 1e-8)).float()


def candidate_weights(quality: list[float] | torch.Tensor, novelty: list[float] | torch.Tensor, coefficient: float) -> torch.Tensor:
    """Apply ExTra's ordering to ES: normalize task quality, then add raw novelty.

    The population replaces GRPO's rollout group; this is an ES/ExTra adaptation.
    Do not re-center or rank the result: doing so would remove a common bonus.
    """
    normalized = z_scores(quality)
    bonus = torch.as_tensor(novelty, dtype=torch.float32)
    if bonus.shape != normalized.shape or not torch.isfinite(bonus).all() or (bonus < 0).any():
        raise ValueError("Novelty must be finite, nonnegative and aligned with quality")
    weights = normalized + coefficient * bonus
    if not torch.isfinite(weights).all():
        raise ValueError("Update weights must be finite")
    return weights


def perturbation_seeds(seed: int, generation: int, population: int) -> list[int]:
    """A pure schedule shared across conditions, independent of runtime RNG use."""
    rng = random.Random(f"es-v1:{seed}:{generation}")
    return rng.sample(range(2**62), population)


class FullWeightES:
    """Reconstruct every candidate from a master, never subtract rounded perturbations."""

    def __init__(self, model: torch.nn.Module, sigma: float, alpha: float, noise_device: str):
        self.parameters = dict(model.named_parameters(remove_duplicate=True))
        self.master = {
            name: p.detach().to(device="cpu", dtype=torch.float32).clone() for name, p in self.parameters.items()
        }
        self.sigma, self.alpha = sigma, alpha
        self.noise_device = torch.device(noise_device)

    def noise(self, name: str, shape: torch.Size, seed: int) -> torch.Tensor:
        """Hash tensor name into a stream seed; regenerate identical layer noise on demand."""
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        tensor_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=self.noise_device).manual_seed(tensor_seed)
        return torch.randn(shape, dtype=torch.float32, device=self.noise_device, generator=generator)

    @torch.no_grad()
    def reconstruct(self, seed: int | None) -> None:
        """Copy a fresh FP32 master-plus-noise value into the inference tensor's dtype."""
        for name, param in self.parameters.items():
            value = self.master[name].to(self.noise_device, copy=True)
            if seed is not None:
                value.add_(self.noise(name, param.shape, seed), alpha=self.sigma)
            param.copy_(value)

    @torch.no_grad()
    def update(self, seeds: list[int], weights: list[float]) -> None:
        """Apply alpha/N times supplied coefficients, without a second normalization."""
        if len(seeds) != len(weights) or not seeds:
            raise ValueError("A nonempty seed is required for each update weight")
        if not torch.isfinite(torch.as_tensor(weights)).all():
            raise ValueError("Update weights must be finite")
        if not any(weights):
            return
        for name, master in self.master.items():
            delta = torch.zeros_like(master, device=self.noise_device)
            for seed, weight in zip(seeds, weights, strict=True):
                if weight:
                    delta.add_(self.noise(name, master.shape, seed), alpha=weight)
            value = master.to(self.noise_device, copy=True)
            value.add_(delta, alpha=self.alpha / len(seeds))
            master.copy_(value)

    def load_master(self, master: dict[str, torch.Tensor]) -> None:
        """Validate shapes and names before restoring an interrupted optimizer."""
        if master.keys() != self.master.keys():
            raise ValueError("Checkpoint parameter names differ")
        for name, tensor in master.items():
            if tensor.shape != self.master[name].shape or tensor.dtype != torch.float32:
                raise ValueError(f"Checkpoint tensor mismatch: {name}")
            self.master[name].copy_(tensor)
        self.reconstruct(None)
