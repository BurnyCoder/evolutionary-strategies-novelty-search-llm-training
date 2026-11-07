"""Independent implementation of Gaussian ES with tie-aware rank shaping.

Math reference: https://arxiv.org/abs/2509.24372 (not its restricted source code).
Rank shaping precedent: https://github.com/uber-research/deep-neuroevolution .
PyTorch seeded generators: https://docs.pytorch.org/docs/stable/generated/torch.Generator.html .
FP32 masters live on the CPU; layer-sized GPU temporaries bound device memory.
"""

from __future__ import annotations

import hashlib
import random

import torch


def centered_ranks(values: list[float]) -> torch.Tensor:
    """Average ranks map to [-.5, .5]; ties carry no arbitrary candidate ordering."""
    scores = torch.as_tensor(values, dtype=torch.float64)
    if not torch.isfinite(scores).all():
        raise ValueError("Fitness must be finite")
    n = len(scores)
    if n < 2:
        return torch.zeros(n)
    # Pairwise comparisons avoid scipy and exactly match average ordinal ranks.
    lower = (scores[:, None] > scores[None, :]).sum(1)
    equal = (scores[:, None] == scores[None, :]).sum(1)
    return ((lower + (equal - 1) / 2) / (n - 1) - 0.5).float()


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
    def update(self, seeds: list[int], fitness: list[float]) -> None:
        """Apply alpha/N times ranked noise; alpha already absorbs inverse sigma."""
        if len(seeds) != len(fitness) or not seeds:
            raise ValueError("A nonempty seed is required for each fitness")
        weights = centered_ranks(fitness).tolist()
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
