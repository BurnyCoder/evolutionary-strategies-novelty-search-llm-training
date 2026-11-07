# All experiments

The current registered study is the paired pretrained pilot. No scratch or diagnostic run is represented as executed.

The directional improvement hypothesis was not supported in this seed: main accuracy did not exceed reward-only ES.

Both trained checkpoints score higher than the original model on this test set; the main-versus-control comparison determines whether novelty adds a benefit. Both also produce more legal expressions on the three/four-number test problems. Legality alone does not mean the target was reached. Both have lower expression validity on the five-number test problems than the original model, so the validity improvement does not transfer uniformly.

| Condition | Test accuracy | 95% interval | Hard accuracy | pass@1 | pass@8 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.29% | 0.10% - 0.54% | 0.10% | 0.12% | 0.89% | complete |
| main | 1.12% | 0.68% - 1.61% | 0.00% | 0.27% | 2.06% | complete |
| reward_only | 1.12% | 0.68% - 1.61% | 0.10% | 0.22% | 1.67% | complete |

Qwen/Qwen2.5-0.5B-Instruct: 24 layers, width 896, 14 query heads and 2 key/value heads; 494,032,768 unique parameter elements in 290 tensors. Architecture and counts come from the pinned pretrained files and the generation-zero checkpoint audit.

| Condition | Parameter update | Objective | Training data / schedule |
| --- | --- | --- | --- |
| Original | Frozen pretrained checkpoint | No optimization here | No training here |
| Main | Full-weight Gaussian ES | Z(task reward) + 1.0 novelty | Registered Countdown split; rotating problems and fixed probes |
| Reward-only | Full-weight Gaussian ES | Z(task reward) | Same problems, perturbation seeds and generation count |

[Full pilot report](experiment-pilot.md). [Deferred studies](../docs/roadmap.md).

ES means comparing random weight changes instead of using backpropagation. Novelty is distance from previous correct answers. Quality is verified arithmetic reward. Accuracy is exact success frequency. pass@k estimates the chance of at least one success among k samples. A confidence interval here describes uncertainty from which test problems were sampled, not uncertainty over training seeds.
