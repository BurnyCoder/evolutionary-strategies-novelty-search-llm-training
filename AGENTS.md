# Project instructions

The user's full research specification governs this project. Preserve its scientific protocol and do not treat a promising interim metric as permission to change the experiment.

- Work only in this repository (or explicitly authorized extra storage), using uv, Python 3.12, the local .venv and ignored .env configuration.
- Start from the source audit in docs/literature.md. Verify important method claims against papers and code; preserve license attribution in NOTICE.
- This project is strictly noncommercial research. Academic/noncommercial upstream code is eligible for reuse when its actual terms cover this scope. Preserve upstream licenses, notices, modification records and redistribution requirements; do not treat a commercial-use restriction as a blanket research-use prohibition.
- Read reports/preregistration.md and configs/pilot.json before changing training or evaluation. Registration was published before model execution. Record any amendment with its timing and reason; never reset an existing experiment deadline.
- The first pilot compares seed 0 full-weight ES plus correctness-gated novelty, matched reward-only ES, and the original model. All correctness, split, archive, rank, replay and deadline tests must pass. Keep calibration and generation budgets matched and distinguish incomplete outcomes.
- Do not expose test prompts to training, choose checkpoints on test scores, change the primary descriptor post hoc, or infer seed robustness from problem-bootstrap intervals.
- Candidate reconstruction starts from FP32 masters, not add/subtract cycles in BF16. Correctness is an explicit Boolean. Score against frozen memory before seeded insertion. Do not use the 0.1 legality reward as the novelty gate.
- Publish and test core changes before running models. Make meaningful commits and feature pull requests; merge authorized PRs autonomously. Never publish secrets or add assistant attribution to commits, PR messages or README.
- Tests and reports use real entry points and artifacts. One report command generates all empirical numbers, tables, plots, model cards, static results and the PDF. Never type empirical scores manually.
- Keep complete timestamped prompt/output logs, atomic checkpoints, RNGs, progress state and the original absolute deadline. Training longer than 30 minutes must remain resumable after process/OS restart.
- Maintain docs/, reports/, the README, paper, model cards and interactive GitHub Pages results. The live dashboard is read-only and loopback-bound. Publish models in one Hugging Face research collection.
- Report positive, negative, failed and inconclusive outcomes with appropriate limits. Paper author is Libor "Burny" Burian; include the current date. The required AI statement belongs in the paper only.
- Finish and publish the first pilot before expanding to the six paired pretrained seeds, diagnostic controls or the separate TinyStories scratch study. Freeze each later methodological amendment before its results exist.
- Do one bounded review unless a serious defect justifies more. Prefer simple library-backed components, useful contextual comments and focused tests to speculative infrastructure.
