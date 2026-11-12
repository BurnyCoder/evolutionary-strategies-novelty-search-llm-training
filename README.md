# Full-weight ES with LLM response novelty

This research project tests whether rewarding **correct but different answers** helps evolution strategies fine-tune a language model. It combines an adapted [ExTra](https://arxiv.org/abs/2606.24994) response-memory mechanism with full-weight optimization inspired by [ES at Scale](https://arxiv.org/abs/2509.24372). Each candidate is a small Gaussian perturbation of every unique model parameter; exact arithmetic grades its Countdown answers, and a frozen embedding model measures novelty among correct answers. The combination is an adaptation, not a reproduction of either paper.

The first local GPU experiment is complete: one main-method seed, one matched reward-only ES seed, and the untrained base model, evaluated within the persisted twelve-hour deadline. Both trained arms started from the original pretrained weights with [Amendment A](reports/preregistration.md#amendment-a-source-aligned-weighting-before-either-trained-condition) active from their first generation. Later multi-seed and scratch studies remain planned. The generated results below and [full report](reports/experiment-pilot.md) retain the positive and negative findings.

A [registered longer continuation](reports/preregistration-continuation.md) extends both existing arms to generation 100 under the same training method, with a fresh disjoint primary evaluation set. An OS interruption stopped its control training, and that attempt's deadline expired. The separately [registered recovery](reports/preregistration-recovery.md) preserves the completed main model and unfinished control state under a new bounded allowance, while retaining the interrupted evidence and original deadlines. The fixed target and analysis do not depend on interim accuracy. See the [recovery commands](docs/continuation.md).

The recovered comparison is now evaluated. Explore the [longer-training results](https://burnycoder.github.io/evolutionary-strategies-novelty-search-llm-training/continuation-seed0-gen100/), [paper](output/pdf/novelty-es-continuation-seed0-gen100.pdf), and [interpretation and limitations](reports/analysis-notes-continuation.md). The first pilot remains a separate release below.

<!-- BEGIN GENERATED CONTINUATION RESULTS -->
The directional improvement hypothesis was not supported in this seed: main accuracy did not exceed reward-only ES. The paired interval includes zero, so this does not establish a performance difference.

Both arms improved over their earlier checkpoints when evaluated on the same fresh problems. Both trained checkpoints score higher than the original model on this test set; the main-versus-control comparison determines whether novelty adds a benefit. Both also produce more legal expressions on the three/four-number test problems. Legality alone does not mean the target was reached.

| Condition | Fresh accuracy | 95% interval | Hard accuracy | pass@1 | pass@8 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.24% | 0.05% - 0.49% | 0.10% | 0.12% | 0.89% | complete |
| main | 7.37% | 6.25% - 8.50% | 0.39% | 2.27% | 10.66% | complete |
| reward_only | 7.71% | 6.54% - 8.89% | 0.88% | 2.47% | 11.95% | complete |

Problem uncertainty conditional on one training seed; excludes seed variability. [Full report](reports/experiment-continuation-seed0-gen100.md).
<!-- END GENERATED CONTINUATION RESULTS -->

<!-- BEGIN GENERATED PILOT RESULTS -->
The directional improvement hypothesis was not supported in this seed: main accuracy did not exceed reward-only ES.

Both trained checkpoints score higher than the original model on this test set; the main-versus-control comparison determines whether novelty adds a benefit. Both also produce more legal expressions on the three/four-number test problems. Legality alone does not mean the target was reached. Both have lower expression validity on the five-number test problems than the original model, so the validity improvement does not transfer uniformly.

| Condition | Test accuracy | 95% interval | Hard accuracy | pass@1 | pass@8 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.29% | 0.10% - 0.54% | 0.10% | 0.12% | 0.89% | complete |
| main | 1.12% | 0.68% - 1.61% | 0.00% | 0.27% | 2.06% | complete |
| reward_only | 1.12% | 0.68% - 1.61% | 0.10% | 0.22% | 1.67% | complete |

Problem uncertainty conditional on one training seed; excludes seed variability. [Full report](reports/experiment-pilot.md).
<!-- END GENERATED PILOT RESULTS -->

![Recorded test accuracy with problem-bootstrap intervals](reports/figures/accuracy.png)

```mermaid
flowchart LR
  D[Disjoint witnessed arithmetic data] --> P[Same rotating prompts and fixed probes]
  M[FP32 master weights] --> N[Seeded full-weight Gaussian candidates]
  P --> G[BF16 model generation]
  N --> G
  G --> Q[Exact AST task reward]
  G --> E[Frozen MiniLM embeddings]
  E --> A[Correct historical and sibling answers]
  A --> F[Add correctness-gated novelty]
  Q --> Z[Population z-score of task reward]
  Z --> F
  F --> R[Weighted Gaussian update]
  R --> M
  M --> C[Atomic checkpoint and live metrics]
  M --> T[Held-out final evaluation]
  T --> O[Generated report, paper and website]
```

## Protocol and evidence

- [Preregistration](reports/preregistration.md): hypothesis, data hashes, primary metric, falsification, timing, controls and analysis.
- [Literature and source audit](docs/literature.md): LLM-specific alternatives, reuse, licenses and paper/code discrepancies.
- [Source correspondence](docs/source-conformance.md): matching equations, retained differences and differential checks against pinned upstream code.
- [Initialization audits](reports/initialization-audits.json) and [final audit](reports/final-audit.json): original pretrained equality, common recorded prompt/noise schedules, grades, coefficients and exported tensors.
- [Hugging Face research collection](https://huggingface.co/collections/BurnyCoder/evolution-strategies-with-llm-novelty-6aeef0a424ee9d502250b8ff): model exports and the raw evidence release.
- [Publication receipt](reports/publication.json) and [remote integrity checks](reports/publication-audit.json): immutable Hub revisions, exact file hashes and verified publication timing.
- [Amended training configuration](configs/pilot.json), [data manifest](data/manifest.json), [roadmap](docs/roadmap.md).

The main endpoint is held-out greedy accuracy of the final scheduled checkpoint. The reward-only condition isolates the added novelty objective. The original checkpoint is the untrained baseline. Paired problem-bootstrap intervals describe uncertainty **conditional on one training seed**. They cannot establish robustness across seeds. Novelty reward, expression-tree diversity and harder-problem accuracy are secondary diagnostics. Results tables and figures are generated from recorded artifacts, including failures and negative findings.

## Installation

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA-capable GPU compatible with the locked PyTorch CUDA 13 build. The initial target is an RTX 5070 laptop with 8 GB VRAM. FP32 master weights stay in system RAM; inference uses BF16 and layer-sized perturbation buffers. The tested dependency versions are in `uv.lock`; no paid cloud service is used.

```powershell
git clone https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training.git
cd evolutionary-strategies-novelty-search-llm-training
uv sync --locked
Copy-Item .env.example .env
uv run pytest
```

User-facing commands (publish and test the core before starting `run`):

```powershell
uv run novelty-es prepare
uv run novelty-es download
uv run novelty-es dashboard
uv run novelty-es run
# The same command resumes the same run and absolute deadline after interruption.
uv run novelty-es report
```

Run the dashboard in a second terminal at `http://127.0.0.1:8765`. The training terminal and JSONL files retain complete prompts and outputs. `.env`, caches, private credentials, intermediate weights and raw run directories are ignored by Git. Models, resumable FP32 checkpoints, complete generation traces and raw evaluations are published separately to one Hugging Face collection. See the [operations guide](docs/operations.md) for checkpoints, recovery, and artifact paths. The [generated report](reports/experiment-pilot.md) and [interactive results](https://BurnyCoder.github.io/evolutionary-strategies-novelty-search-llm-training/) show all conditions and evaluation coverage.

## Method and reproducibility

Qwen2.5-0.5B-Instruct and all-MiniLM-L6-v2 are pinned by commit. The synthetic generator creates solvable arithmetic problems and prevents permutations or alternative targets for a number multiset from crossing splits. A restricted AST evaluator uses exact rational arithmetic and requires every supplied number exactly once. Task fitness averages both the rotating training batch and fixed training probes. Only exact correctness enables novelty; the smaller legality reward does not.

Each generation evaluates independent seeded Gaussian candidates. Task rewards are standardized across the population following ES at Scale. The main condition adds correctness-gated nearest-neighbor novelty afterwards, following ExTra's normalization order; the control uses standardized task reward alone. No second normalization or ranking removes the bonus. The coefficient uses ExTra's reported default in normalized-quality units, without a claim that it is optimal for ES. Candidate reconstruction starts from the FP32 master to avoid cumulative BF16 add/subtract drift. Checkpoints preserve weights, RNGs, archive, cursor, configuration and the original deadline. The same seed schedules support a paired comparison; validation never selects the primary checkpoint.

[Amendment A](reports/preregistration.md#amendment-a-source-aligned-weighting-before-either-trained-condition) supersedes the original centered-rank rule. It was published after calibration and partial original-model evaluation, before either trained condition began. Both trained arms use the amended rule from their first generation and start from the original pretrained weights. The source correspondence audit distinguishes matching equations from intentional numerical and task adaptations.

Technologies: Python, PyTorch, Transformers, Sentence Transformers, NumPy, a standard-library local HTTP dashboard, Matplotlib, and ReportLab. The default pipeline is `prepare → calibrate → original → main → reward-only → held-out evaluation → report`. Details, deviations and experiment outcomes live in `docs/` and `reports/` rather than duplicated prose. See [NOTICE](NOTICE) for adapted code attribution and [LICENSE](LICENSE) for source terms.
