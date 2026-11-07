# Copyright 2025 ExTra Authors
# Modifications Copyright 2026 Libor Burian
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
"""Correctness-filtered adaptation of ExTra's embedding memory.

Upstream: allen4747/extra at ea53e68036356f8f276e261e886fbbc102633be3,
verl/recipe/extra/curiosity_memory.py (add_rollout_embeddings).
https://arxiv.org/html/2606.24994v1, Section 4.1 and Appendix B.3.

Changes: candidate-model siblings, explicit Boolean correctness mask, no warmup
or group normalization, separate frozen scoring and seeded FIFO insertion.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


class CorrectMemory:
    """Store only correct embeddings per probe, with reproducible bounded insertion."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.history: dict[int, torch.Tensor] = {}
        self.rng = torch.Generator().manual_seed(seed)

    @staticmethod
    def _check(embeddings: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        """Refuse reward-as-gate mistakes and nonfinite model embeddings."""
        if correct.dtype != torch.bool:
            raise TypeError("Pass Boolean exact correctness, never reward > 0")
        if embeddings.ndim != 3 or embeddings.shape[:2] != correct.shape:
            raise ValueError("Expected [candidate, probe, dimension] and [candidate, probe]")
        if not torch.isfinite(embeddings).all() or (embeddings.norm(dim=-1) == 0).any():
            raise ValueError("Embeddings must be finite and nonzero")
        return F.normalize(embeddings.detach().float().cpu(), dim=-1)

    def score(self, embeddings: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        """Score against immutable history and all other correct candidate siblings."""
        normalized = self._check(embeddings, correct)
        rewards = torch.zeros(correct.shape, dtype=torch.float32)
        for probe in range(correct.shape[1]):
            indices = correct[:, probe].cpu().nonzero().flatten()
            if not len(indices):
                continue
            current = normalized[indices, probe]
            history = self.history.get(probe, current[:0])
            pool = torch.cat((history, current))
            similarities = current @ pool.T
            # Sibling position follows history; masking never removes other duplicates.
            similarities[torch.arange(len(indices)), len(history) + torch.arange(len(indices))] = -torch.inf
            nearest = similarities.max(dim=1).values
            nearest = torch.where(torch.isneginf(nearest), 0.0, nearest).clamp(-1.0, 1.0)
            rewards[indices, probe] = 1.0 - nearest
        return rewards

    def append(self, embeddings: torch.Tensor, correct: torch.Tensor) -> None:
        """After scoring all candidates, append only correct answers in seeded order."""
        normalized = self._check(embeddings, correct)
        for probe in range(correct.shape[1]):
            indices = correct[:, probe].cpu().nonzero().flatten()
            if not len(indices):
                continue
            indices = indices[torch.randperm(len(indices), generator=self.rng)]
            current = normalized[indices, probe]
            history = self.history.get(probe, current[:0])
            self.history[probe] = torch.cat((history, current))[-self.capacity :].clone()

    def state_dict(self) -> dict:
        """Clone state so later insertions cannot modify a pending checkpoint."""
        return {
            "capacity": self.capacity,
            "history": {k: v.clone() for k, v in self.history.items()},
            "rng": self.rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore both archive contents and insertion order RNG."""
        if state["capacity"] != self.capacity:
            raise ValueError("Archive capacity changed on resume")
        self.history = {k: v.clone() for k, v in state["history"].items()}
        self.rng.set_state(state["rng"])
