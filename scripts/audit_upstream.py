"""CPU-only differential audit against reviewed, commit-pinned source components.

Run with ``uv run python scripts/audit_upstream.py``. Downloaded references and
their licenses remain in ignored tmp/, not in this project's source distribution.
ES at Scale's Academic Public License permits this noncommercial research use.
The comparison covers shared numerical kernels, not equivalence of full trainers.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from novelty_es.novelty import CorrectMemory
from novelty_es.optimizer import z_scores


def reference(repository: str, revision: str, source: str, license_name: str, name: str):
    """Load only the specific source already reviewed, retaining its exact bytes/license."""
    directory = Path("tmp/source-audit") / name
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename in (source, license_name):
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/{filename}"
        content = urllib.request.urlopen(url, timeout=30).read()
        target = directory / Path(filename).name
        target.write_bytes(content)
        files[filename] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    spec = importlib.util.spec_from_file_location(name, directory / Path(source).name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"repository": repository, "revision": revision, "files": files}


def main():
    """Compare common nonempty-memory cases; dedicated unit tests cover empty history."""
    torch.set_num_threads(1)
    extra, extra_provenance = reference(
        "allen4747/extra", "ea53e68036356f8f276e261e886fbbc102633be3",
        "verl/recipe/extra/curiosity_memory.py", "LICENSE", "extra",
    )
    es, es_provenance = reference(
        "VsonicV/es-at-scale", "574a9d134da1ffce2a8bb812019899e5c96b588a",
        "es_at_scale/utils/reward_shaping.py", "LICENSE.txt", "es_at_scale",
    )
    rng = torch.Generator().manual_seed(20261107)
    error, z_error, comparisons = 0.0, 0.0, 0
    trials = 64
    for trial in range(trials):
        count, dimension, probes = 1 + trial % 30, 11, 3
        embeddings = torch.randn(count, probes, dimension, generator=rng)
        correct = torch.rand(count, probes, generator=rng) > 0.35
        ours = CorrectMemory(6, 0)
        for probe in range(probes):
            ours.history[probe] = F.normalize(torch.randn(1 + trial % 6, dimension, generator=rng), dim=-1)
        observed = ours.score(embeddings, correct)
        for probe in range(probes):
            indices = correct[:, probe].nonzero().flatten()
            if not len(indices):
                continue
            upstream = extra.CuriosityMemory(0, max_rollouts_per_prompt=6, min_history_before_reward=0)
            upstream.rollout_memory[(probe,)] = deque(ours.history[probe].unbind(), maxlen=6)
            # Correctness filtering reconciles the helper with the paper's memory specification.
            expected = upstream.add_rollout_embeddings(
                [(probe,)] * len(indices), embeddings[indices, probe], normalize_per_group=False,
            )
            error = max(error, float((observed[indices, probe] - expected).abs().max()))
            comparisons += len(indices)
            torch.testing.assert_close(observed[indices, probe], expected, atol=1e-6, rtol=0)
        assert torch.equal(observed[~correct], torch.zeros_like(observed[~correct]))
        values = torch.rand(count, generator=rng, dtype=torch.float64).numpy()
        population = {i: {"avg_reward": value} for i, value in enumerate(values)}
        with contextlib.redirect_stdout(io.StringIO()):
            expected_scores = es.z_score(population, np.mean(values), np.std(values))
        expected = torch.tensor([expected_scores[i]["norm_reward"] for i in population], dtype=torch.float32)
        actual = z_scores(values.tolist())
        z_error = max(z_error, float((actual - expected).abs().max()))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "device": "cpu", "seed": 20261107,
        "trials": trials, "novelty_values_compared": comparisons,
        "max_absolute_novelty_error": error, "max_absolute_z_score_error": z_error,
        "passed": True, "sources": [extra_provenance, es_provenance],
        "scope": "Shared cosine kernel with correct-only input, nonempty history, no warmup/group rescaling; ES population z-scores. Not full-trainer equivalence.",
    }
    path = Path("reports/source-conformance.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
