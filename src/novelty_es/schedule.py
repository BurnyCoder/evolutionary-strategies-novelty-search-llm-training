"""Pure schedules make candidate/prompt allocation independent of run condition."""

import math
import random


class PromptSchedule:
    """Hold out fixed training probes and cycle deterministic shuffled training epochs."""

    def __init__(self, count: int, seed: int, probe_count: int, rotating_count: int):
        indices = list(range(count))
        random.Random(f"probes-v1:{seed}").shuffle(indices)
        self.probes = indices[:probe_count]
        self.pool = indices[probe_count:]
        self.seed, self.rotating_count = seed, rotating_count
        if rotating_count > len(self.pool):
            raise ValueError("Rotating batch exceeds non-probe training pool")

    def indices(self, generation: int) -> list[int]:
        """Use complete nonoverlapping batches; skip a short tail to avoid duplicates."""
        batches = len(self.pool) // self.rotating_count
        epoch, batch = divmod(generation, batches)
        pool = self.pool.copy()
        random.Random(f"epoch-v1:{self.seed}:{epoch}").shuffle(pool)
        start = batch * self.rotating_count
        return pool[start : start + self.rotating_count] + self.probes


def freeze_generation_count(
    candidate_seconds: list[float],
    update_seconds: float,
    checkpoint_seconds: float,
    validation_seconds: float,
    population: int,
    validation_every: int,
    allowance: float,
    fraction: float,
    cap: int,
) -> tuple[int, float]:
    """Apply the registered conservative timing rule without observing test accuracy."""
    estimate = population * max(candidate_seconds) + update_seconds + checkpoint_seconds
    estimate += validation_seconds / validation_every
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError("Invalid calibration timing")
    count = min(cap, math.floor(fraction * allowance / estimate))
    if count < 1:
        raise ValueError("No full generation fits the registered allowance")
    return count, estimate
