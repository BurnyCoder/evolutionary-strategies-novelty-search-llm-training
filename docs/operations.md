# Running, resuming and inspecting the pilot

Run all commands from the repository root. `uv sync --locked` installs the recorded Python 3.12 environment. Copy `.env.example` to `.env`; the local cache path does not replace your Hugging Face authentication location. The project never prints authentication tokens.

`uv run novelty-es prepare` verifies existing data hashes or constructs the registered synthetic data if absent. `uv run novelty-es download` downloads the pinned model and embedding revisions without running them. Tests require no model inference or network calls: `uv run pytest`.

`uv run novelty-es dashboard` serves read-only local metrics on `127.0.0.1:8765`. Start this in a separate terminal. `uv run novelty-es run` starts the calibration clock, freezes the common generation count from measured timings, evaluates the original checkpoint, trains the main and reward-only arms, evaluates final checkpoints, and generates reports. The same run command resumes the same directory and deadline. Do not run two writers on the same directory. The CLI refuses dirty source before a new experiment.

## Artifact map

| Path under `artifacts/pilot` | Role |
|---|---|
| `experiment.json` | Registered and effective configuration, provenance, absolute deadline, phase state and failures |
| `calibration.json` | Measured costs and frozen common generation count |
| `events.jsonl` | Timestamped complete prompts, partial decoding texts, complete responses and phase events |
| `{condition}/checkpoint.pt` | Last fully committed generation, FP32 masters, RNG state, archive, cursor and counters |
| `{condition}/pending.pt` | Completed candidates in the unfinished generation |
| `{condition}/metrics.json` | Committed generation curves |
| `{condition}/evaluation.json` | Raw per-problem and per-sample outcomes, resumable at batch boundaries |
| `{condition}/model/` | Hugging Face-compatible final export and generated model card |
| `summary.json` | Canonical estimates, intervals, completion flags and selected fixed examples |

Both `main` and `reward_only` compute probe embeddings. Archive histories differ because they arise from different policies; the cost categories and candidate/prompt schedules are matched. Generation counts, not realized wall time or output token lengths, are matched. All comparisons disclose token and runtime differences.

Interrupted generations restore the last master and replay only unfinished candidates before applying the update once. A crash during checkpoint serialization leaves the previous checkpoint valid. A reboot does not restore a process automatically: restart `uv run novelty-es run`; downtime still consumes the original allowance. For a long unattended run, keep the laptop powered and prevent sleep using the operating system's normal power settings.

`uv run novelty-es report` regenerates Markdown, figures, the interactive static page, model-card text and `output/pdf/novelty-es-pilot.pdf`. The PDF must be visually inspected before final delivery. `uv run novelty-es publish` uploads model exports and a narrowly allowlisted set of raw evidence to one Hugging Face collection. GitHub publication uses normal meaningful commits and feature pull requests. GitHub Pages deploys `docs/site` through the included workflow.

Reports distinguish positive novelty from a bonus that actually changes candidate rank weights. This is a post-hoc descriptive calculation on the observed populations, documented in [analysis notes](../reports/analysis-notes.md); it does not replay an alternative training trajectory. Token counts distinguish committed training/evaluation rows from all final response events in the trace, including logged abandoned work. Missing evaluation denominators and the limitations of degenerate bootstrap intervals remain explicit.

Extra local storage is available at `D:\AI\evolutionary-strategies-novelty-search-llm-training`. The active pilot keeps its existing paths; completed, stable artifacts can be copied there with SHA-256 verification. Do not move active checkpoints or open them for a long backup copy while the training writer may atomically replace them. Use the extra disk for subsequent model archives and larger studies after the first report is published.

## Failure handling

Out-of-memory batch fallback is permitted only by the registered mechanical rule. Other failures are recorded and investigated; no silent seed changes or schedule extensions. Unfinished conditions may have partial evaluations but cannot support the registered final-checkpoint comparison. Model execution is local and uses `trust_remote_code=False`. Checkpoints are loaded with PyTorch's weights-only mechanism and must come from this project. No generated expression is evaluated as Python code.

## Implementation sources

The [literature audit](literature.md) records mathematical and licensing provenance. API references appear in module docstrings. Atomic replacement uses Python `os.replace`; deterministic noise uses explicit PyTorch generators; inference and chat templates use the pinned Transformers release; normalized embeddings use Sentence Transformers. Rendering uses Matplotlib and ReportLab. The dashboard uses the Python standard library with an explicit route allowlist and a loopback-only bind.
