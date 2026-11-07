# Pilot experiment

Libor "Burny" Burian · 2026-11-07

The directional improvement hypothesis was not supported in this seed: main accuracy did not exceed reward-only ES.

Paired difference: 0.000 percentage points; 95% interval [-0.244, 0.244]; 2048 paired test problems.

Problem uncertainty conditional on one training seed; excludes seed variability.

Both trained checkpoints score higher than the original model on this test set; the main-versus-control comparison determines whether novelty adds a benefit. Both also produce more legal expressions on the three/four-number test problems. Legality alone does not mean the target was reached. Both have lower expression validity on the five-number test problems than the original model, so the validity improvement does not transfer uniformly.

Percentile bootstrap intervals can collapse at observed boundaries or when paired outcomes are identical. A zero-width interval does not prove zero population error or equivalence on unseen problems.

| Condition | Test accuracy | 95% interval | Hard accuracy | pass@1 | pass@8 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.29% | 0.10% - 0.54% | 0.10% | 0.12% | 0.89% | complete |
| main | 1.12% | 0.68% - 1.61% | 0.00% | 0.27% | 2.06% | complete |
| reward_only | 1.12% | 0.68% - 1.61% | 0.10% | 0.22% | 1.67% | complete |

original: greedy problems test 2048/2048, hard 1024/1024, probes 16/16; sampled complete problems 256/256. main: greedy problems test 2048/2048, hard 1024/1024, probes 16/16; sampled complete problems 256/256. reward_only: greedy problems test 2048/2048, hard 1024/1024, probes 16/16; sampled complete problems 256/256.

| Comparison | Difference (pp) | Paired 95% interval (pp) | Problems |
| --- | --- | --- | --- |
| Main - reward-only (primary) | 0.000 | [-0.244, 0.244] | 2048 |
| main - original (secondary) | 0.830 | [0.438, 1.270] | 2048 |
| reward_only - original (secondary) | 0.830 | [0.391, 1.270] | 2048 |

Comparisons with the original checkpoint are descriptive secondary estimates with unadjusted intervals.

![Accuracy](figures/accuracy.png)

![Training](figures/training.png)

## Probe performance, validity and resources

| Condition | Probe accuracy | Test validity | Hard validity | Train tokens | Eval tokens | Train hours |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.00% | 29.30% | 10.64% | 0 | 385121 | 0.000 |
| main | 0.00% | 76.12% | 2.05% | 798675 | 217202 | 1.688 |
| reward_only | 0.00% | 79.79% | 3.52% | 814802 | 229510 | 1.662 |

Train tokens count only candidates in committed generations; evaluation tokens count persisted evaluation responses. Calibration, validation, interrupted batches and replay costs are excluded from those columns. Training hours include phase overhead and checkpointing. Probes are training examples, not held-out data.

## Novelty intervention diagnostic

Descriptive diagnostic: compare the main arm's Z(Q) + 1.0 N with Z(Q) on its recorded populations. The control computes novelty but applies coefficient zero, so its novelty-induced weight changes are zero. This local comparison does not simulate the counterfactual training trajectory. main: positive novelty in 15 of 570 candidates; changed update weights in 9 of 19 generations (14 candidate weights). reward_only: positive novelty in 13 of 570 candidates; changed update weights in 0 of 19 generations (0 candidate weights).

See [analysis notes](analysis-notes.md) for when this descriptive check was added.

## Within-problem expression diversity

| Condition | Greedy correct structures | Complete sampled problems | Mean correct structures / sampled problem |
| --- | --- | --- | --- |
| original | 6 | 256 | 0.0156 |
| main | 23 | 256 | 0.0391 |
| reward_only | 23 | 256 | 0.0312 |

Greedy structure counts equal successful problems, since each problem has one response. Sampled diversity counts distinct correct canonical ASTs within each problem, then averages over complete sampled groups, including zeros. This is syntactic diversity; algebraically equivalent expressions may count separately.

## Method and scope

See the [preregistration](preregistration.md), [configuration](../configs/pilot.json), [data hashes](../data/manifest.json), and [literature audit](../docs/literature.md). This is an adaptation of [ExTra](https://arxiv.org/abs/2606.24994) and [ES at Scale](https://arxiv.org/abs/2509.24372), not a reproduction. The archive gate uses exact correctness; positive legality reward is insufficient. Validation does not choose the final checkpoint. Amendment A replaces combined-reward ranks with population-z-scored task reward plus post-normalization novelty. It was published after calibration and partial original-model evaluation, before either trained condition. Both trained arms start from original pretrained weights using the amended rule from generation one.

Qwen/Qwen2.5-0.5B-Instruct: 24 layers, width 896, 14 query heads and 2 key/value heads; 494,032,768 unique parameter elements in 290 tensors. Architecture and counts come from the pinned pretrained files and the generation-zero checkpoint audit.

## Recorded measurements

All values below come from persisted outcomes. Missing conditions remain missing. Population training accuracy is distinct from unperturbed master validation and held-out test accuracy.

```json
{
  "methods": {
    "original": {
      "training_complete": true,
      "evaluation_seconds": 3060.1390517000036,
      "complete": true,
      "test": {
        "n": 2048,
        "expected": 2048,
        "complete": true,
        "accuracy": 0.0029296875,
        "interval": [
          0.0009765625,
          0.00537109375
        ],
        "validity": 0.29296875,
        "reward": 0.03193359375,
        "generated_tokens": 95929,
        "truncated": 529,
        "correct_expression_structures": 6
      },
      "hard": {
        "n": 1024,
        "expected": 1024,
        "complete": true,
        "accuracy": 0.0009765625,
        "interval": [
          0.0,
          0.0029296875
        ],
        "validity": 0.1064453125,
        "reward": 0.0115234375,
        "generated_tokens": 59320,
        "truncated": 333,
        "correct_expression_structures": 1
      },
      "probes": {
        "n": 16,
        "expected": 16,
        "complete": true,
        "accuracy": 0.0,
        "interval": [
          0.0,
          0.0
        ],
        "validity": 0.25,
        "reward": 0.025,
        "generated_tokens": 823,
        "truncated": 5,
        "correct_expression_structures": 0
      },
      "sampled": {
        "complete_problems": 256,
        "expected": 256,
        "generated_tokens": 229049,
        "partial_problems": 0,
        "pass1": 0.001220703125,
        "pass8": 0.008854166666666666,
        "mean_unique_correct_structures": 0.015625
      },
      "training": {},
      "training_observations": {
        "responses": 0,
        "generated_tokens": 0,
        "truncated": 0,
        "weight_diagnostic": {
          "generations_observed": 0,
          "candidates_observed": 0,
          "candidates_positive_novelty": 0,
          "generations_positive_novelty": 0,
          "generations_weight_changed": 0,
          "candidates_weight_changed": 0,
          "largest_absolute_weight_change": 0.0
        }
      },
      "training_seconds": 0.0
    },
    "main": {
      "training_complete": true,
      "evaluation_seconds": 1940.7667252999672,
      "complete": true,
      "test": {
        "n": 2048,
        "expected": 2048,
        "complete": true,
        "accuracy": 0.01123046875,
        "interval": [
          0.0068359375,
          0.01611328125
        ],
        "validity": 0.76123046875,
        "reward": 0.08623046875,
        "generated_tokens": 41525,
        "truncated": 0,
        "correct_expression_structures": 23
      },
      "hard": {
        "n": 1024,
        "expected": 1024,
        "complete": true,
        "accuracy": 0.0,
        "interval": [
          0.0,
          0.0
        ],
        "validity": 0.0205078125,
        "reward": 0.00205078125,
        "generated_tokens": 23091,
        "truncated": 0,
        "correct_expression_structures": 0
      },
      "probes": {
        "n": 16,
        "expected": 16,
        "complete": true,
        "accuracy": 0.0,
        "interval": [
          0.0,
          0.0
        ],
        "validity": 0.875,
        "reward": 0.08750000000000001,
        "generated_tokens": 306,
        "truncated": 0,
        "correct_expression_structures": 0
      },
      "sampled": {
        "complete_problems": 256,
        "expected": 256,
        "generated_tokens": 152280,
        "partial_problems": 0,
        "pass1": 0.002685546875,
        "pass8": 0.020572916666666666,
        "mean_unique_correct_structures": 0.0390625
      },
      "training": {
        "generation": 19,
        "generations": 19,
        "complete": true,
        "generated_tokens": 798675,
        "responses": 27360
      },
      "training_observations": {
        "responses": 27360,
        "generated_tokens": 798675,
        "truncated": 1999,
        "weight_diagnostic": {
          "generations_observed": 19,
          "candidates_observed": 570,
          "candidates_positive_novelty": 15,
          "generations_positive_novelty": 10,
          "generations_weight_changed": 9,
          "candidates_weight_changed": 14,
          "largest_absolute_weight_change": 0.0625
        }
      },
      "training_seconds": 6078.265563964844
    },
    "reward_only": {
      "training_complete": true,
      "evaluation_seconds": 1949.0018551001413,
      "complete": true,
      "test": {
        "n": 2048,
        "expected": 2048,
        "complete": true,
        "accuracy": 0.01123046875,
        "interval": [
          0.0068359375,
          0.01611328125
        ],
        "validity": 0.7978515625,
        "reward": 0.08989257812500001,
        "generated_tokens": 42464,
        "truncated": 0,
        "correct_expression_structures": 23
      },
      "hard": {
        "n": 1024,
        "expected": 1024,
        "complete": true,
        "accuracy": 0.0009765625,
        "interval": [
          0.0,
          0.0029296875
        ],
        "validity": 0.03515625,
        "reward": 0.00439453125,
        "generated_tokens": 23582,
        "truncated": 0,
        "correct_expression_structures": 1
      },
      "probes": {
        "n": 16,
        "expected": 16,
        "complete": true,
        "accuracy": 0.0,
        "interval": [
          0.0,
          0.0
        ],
        "validity": 0.8125,
        "reward": 0.08125000000000002,
        "generated_tokens": 316,
        "truncated": 0,
        "correct_expression_structures": 0
      },
      "sampled": {
        "complete_problems": 256,
        "expected": 256,
        "generated_tokens": 163148,
        "partial_problems": 0,
        "pass1": 0.002197265625,
        "pass8": 0.016666666666666666,
        "mean_unique_correct_structures": 0.03125
      },
      "training": {
        "generation": 19,
        "generations": 19,
        "complete": true,
        "generated_tokens": 814802,
        "responses": 27360
      },
      "training_observations": {
        "responses": 27360,
        "generated_tokens": 814802,
        "truncated": 2093,
        "weight_diagnostic": {
          "generations_observed": 19,
          "candidates_observed": 570,
          "candidates_positive_novelty": 13,
          "generations_positive_novelty": 9,
          "generations_weight_changed": 0,
          "candidates_weight_changed": 0,
          "largest_absolute_weight_change": 0.0
        }
      },
      "training_seconds": 5984.517926454544
    }
  },
  "elapsed_seconds": 20151.339267730713,
  "calibration": {
    "generations": 19,
    "estimated_generation_seconds": 676.9900205400081,
    "candidate_seconds": [
      22.129881900000328,
      4.769208300000173,
      4.601129599999695
    ],
    "update_seconds": 1.6471234999989974,
    "checkpoint_seconds": 5.704658999999083,
    "validation_seconds": 28.708905200001027,
    "batch_size": 16,
    "frozen_at": 1794023906.2766497
  },
  "logged_generation_work": [
    {
      "condition": "calibration",
      "phase": "calibration",
      "responses": 144,
      "generated_tokens": 4602
    },
    {
      "condition": "calibration",
      "phase": "calibration_validation",
      "responses": 64,
      "generated_tokens": 3018
    },
    {
      "condition": "calibration",
      "phase": "calibration_warmup",
      "responses": 16,
      "generated_tokens": 985
    },
    {
      "condition": "main",
      "phase": "evaluation",
      "responses": 7184,
      "generated_tokens": 217202
    },
    {
      "condition": "main",
      "phase": "train",
      "responses": 27360,
      "generated_tokens": 798675
    },
    {
      "condition": "main",
      "phase": "validation",
      "responses": 256,
      "generated_tokens": 5151
    },
    {
      "condition": "original",
      "phase": "evaluation",
      "responses": 7200,
      "generated_tokens": 385831
    },
    {
      "condition": "reward_only",
      "phase": "evaluation",
      "responses": 7184,
      "generated_tokens": 229510
    },
    {
      "condition": "reward_only",
      "phase": "train",
      "responses": 27360,
      "generated_tokens": 814802
    },
    {
      "condition": "reward_only",
      "phase": "validation",
      "responses": 256,
      "generated_tokens": 5142
    }
  ]
}
```

## Failures, deviations and learnings

```json
{
  "failures": [
    {
      "time": 1794026144.5639172,
      "phase": "calibration",
      "exception": "PermissionError",
      "message": "[WinError 5] Access is denied: 'artifacts\\\\pilot\\\\original\\\\evaluation.json.tmp' -> 'artifacts\\\\pilot\\\\original\\\\evaluation.json'"
    }
  ],
  "protocol_amendments": [
    {
      "id": "A",
      "applied_at": 1794027075.1466103,
      "phase_before": "calibration",
      "registration_revision": "158becd",
      "implementation_revision": "9f8d30ef0103e3532ebefeafe7f55b8f8d163a80",
      "reason": "User-requested source alignment before either trained condition; ExTra post-normalization addition and ES at Scale population z-scores",
      "initialization": "Both arms start from original pretrained weights; no prior training checkpoints exist",
      "original_model_evaluation_already_observed": true,
      "changes": {
        "novelty_weight": {
          "before": 0.25,
          "after": 1.0
        },
        "weighting": {
          "before": null,
          "after": "zscore_task_plus_novelty_v1"
        }
      },
      "absolute_deadline": 1794067007.1504683
    }
  ],
  "execution_repairs": [
    {
      "time": 1794026373.4031878,
      "revision": "4cbc4db74b77570c4b66793a544e7f02dda0287b",
      "change": "Bounded retry for Windows atomic replacement locks; no numerical training changes",
      "resume_from": "last committed original-model evaluation batch",
      "absolute_deadline": 1794067007.1504683
    }
  ],
  "recovery_audit": {
    "checked_at": "2026-11-07T05:41:15.301+00:00",
    "condition": "original",
    "split": "sampled",
    "before_sha256": "10f4d15e6946fa25b09bf78a635003b3a6f4de637a26f4b05769485a20f52af2",
    "before_rows": 2128,
    "after_rows_at_check": 3136,
    "replayed_batch_start": 2112,
    "replayed_batch_rows": 16,
    "all_prior_and_replayed_generation_fields_equal": true,
    "excluded_field": "batch_seconds",
    "absolute_deadline": 1794067007.1504683,
    "scope": "Observed storage-failure recovery; not a claim about arbitrary restarts or environments"
  },
  "initial_code_revision": "ddcd928c1392fa2eb5beaff5b7b6f192dfa04817",
  "training_code_revision": "9f8d30ef0103e3532ebefeafe7f55b8f8d163a80"
}
```

The first audit reconciled ExTra's correctness-filtered paper specification with its unfiltered embedding-memory insertion. Explicit Boolean masking and separate score/append operations implement the intended experiment. FP32 master reconstruction avoids repeated BF16 perturbation subtraction. Tests exercise interrupted candidate replay, archive RNG restoration, deadline preservation and exact arithmetic. Hyperparameters are preregistered choices, not claimed optima. Further training requires reviewing this pilot.

## Limitations

One seed; synthetic arithmetic; embedding-proxy and probe-overfitting risks; no general-language benchmark; descriptive secondary analyses. Bootstrap intervals omit seed variability. CPU tests, reporting, audits and backups overlapped training, so timing is not an isolated systems benchmark.
