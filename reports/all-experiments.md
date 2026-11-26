# All experiments

Each study below is regenerated from its own raw records. Probes are training data; sampled pass@k always uses the original test subset.

## pilot

The directional improvement hypothesis was not supported in this seed: main accuracy did not exceed reward-only ES. The paired interval includes zero, so this does not establish a performance difference.

| Condition | Test greedy accuracy | 95% interval | Hard greedy accuracy | Original-test pass@1 | Original-test pass@8 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.29% | 0.10% - 0.54% | 0.10% | 0.12% | 0.89% | complete |
| main | 1.12% | 0.68% - 1.61% | 0.00% | 0.27% | 2.06% | complete |
| reward_only | 1.12% | 0.68% - 1.61% | 0.10% | 0.22% | 1.67% | complete |

Both trained checkpoints score higher than the original model on this test set; the main-versus-control comparison determines whether novelty adds a benefit. Both also produce more legal expressions on the three/four-number test problems. Legality alone does not mean the target was reached. Both have lower expression validity on the five-number test problems than the original model, so the validity improvement does not transfer uniformly.

[Full report](experiment-pilot.md).

The historical scorer accepted one complete answer block even when an extra delimiter appeared. Applying the balanced-tag reading to the saved training responses changes 9 novelty-arm and 11 control-arm rewards in this training stage, including 0 changes to fixed-probe novelty eligibility. On the saved final outputs the balanced-tag difference is +0.000 percentage points (paired 95% interval -0.244 to +0.244). The novelty checkpoint alone solved 3 problems and the control alone solved 3. The post-hoc two-sided exact McNemar p-value is 1. Both decoding paths inherit a repetition penalty of 1.1. The measured checkpoints reflect the historical rewards. The outcomes of training with balanced tags are unknown. [Audit](https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training/blob/main/reports/reviews/scientific-audit-2026-11-26.md).

## continuation-seed0-gen100

The directional improvement hypothesis was not supported in this seed: main accuracy did not exceed reward-only ES. The paired interval includes zero, so this does not establish a performance difference.

| Condition | Fresh greedy accuracy | 95% interval | Hard greedy accuracy | Original-test pass@1 | Original-test pass@8 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.24% | 0.05% - 0.49% | 0.10% | 0.12% | 0.89% | complete |
| main | 7.37% | 6.25% - 8.50% | 0.39% | 2.27% | 10.66% | complete |
| reward_only | 7.71% | 6.54% - 8.89% | 0.88% | 2.47% | 11.95% | complete |

Both arms improved over their earlier checkpoints when evaluated on the same fresh problems. Both trained checkpoints score higher than the original model on this test set; the main-versus-control comparison determines whether novelty adds a benefit. Both also produce more legal expressions on the three/four-number test problems. Legality alone does not mean the target was reached.

[Full report](experiment-continuation-seed0-gen100.md).

The historical scorer accepted one complete answer block even when an extra delimiter appeared. Applying the balanced-tag reading to the saved training responses changes 39 novelty-arm and 37 control-arm rewards in this training stage, including 1 changes to fixed-probe novelty eligibility. On the saved final outputs the balanced-tag difference is -0.342 percentage points (paired 95% interval -1.025 to +0.342). The novelty checkpoint alone solved 21 problems and the control alone solved 28. The post-hoc two-sided exact McNemar p-value is 0.3916. Both decoding paths inherit a repetition penalty of 1.1. The measured checkpoints reflect the historical rewards. The outcomes of training with balanced tags are unknown. [Audit](https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training/blob/main/reports/reviews/scientific-audit-2026-11-26.md).

ES evaluates random parameter perturbations and weights their update by measured reward. Main adds correct-response novelty; control uses task reward alone. The original pretrained checkpoint is frozen. Both trained arms use the same rotating training problems and fixed probes. There are no executed scratch, random-search or archive-ablation studies in this release.

[Method and architecture](../README.md). [Scientific evidence and limitations](reviews/scientific-audit-2026-11-26.md).
