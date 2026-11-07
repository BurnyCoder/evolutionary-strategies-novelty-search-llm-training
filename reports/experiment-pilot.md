# Pilot experiment

Libor "Burny" Burian · 2026-11-07

No completed paired primary endpoint is available.

The registered paired endpoint is incomplete or not yet available.

Problem uncertainty conditional on one training seed; excludes seed variability.

| Condition | Test accuracy | 95% interval | Hard accuracy | pass@1 | pass@8 | Status |
| --- | --- | --- | --- | --- | --- | --- |

![Accuracy](figures/accuracy.png)

![Training](figures/training.png)

## Method and scope

See the [preregistration](preregistration.md), [configuration](../configs/pilot.json), [data hashes](../data/manifest.json), and [literature audit](../docs/literature.md). This is an adaptation of [ExTra](https://arxiv.org/abs/2606.24994) and [ES at Scale](https://arxiv.org/abs/2509.24372), not a reproduction. The archive gate uses exact correctness; positive legality reward is insufficient. Validation does not choose the final checkpoint.

## Recorded measurements

All values below come from persisted outcomes. Missing conditions remain missing. Population training accuracy is distinct from unperturbed master validation and held-out test accuracy.

```json
{
  "methods": {},
  "elapsed_seconds": 0,
  "calibration": {}
}
```

## Failures, deviations and learnings

[]

The first audit reconciled ExTra's correctness-filtered paper specification with its unfiltered embedding-memory insertion. Explicit Boolean masking and separate score/append operations implement the intended experiment. FP32 master reconstruction avoids repeated BF16 perturbation subtraction. Tests exercise interrupted candidate replay, archive RNG restoration, deadline preservation and exact arithmetic. Hyperparameters are preregistered choices, not claimed optima. Further training requires reviewing this pilot.

## Limitations

One seed; synthetic arithmetic; embedding-proxy and probe-overfitting risks; no general-language benchmark; descriptive secondary analyses. Bootstrap intervals omit seed variability.
