# Independent corrections using existing evidence

Libor "Burny" Burian · 2026-11-12

This revision independently checks the existing pilot and continuation. It does not train, resume, generate model responses or introduce another experimental run. Earlier audits are preserved as historical assertions. The [claim ledger](claim-ledger.json) records claim locations, type, primary evidence, method, disposition and correction; the [file inventory](claim-inventory.json) maps the reviewed surfaces to those claims. A verified disposition is bounded by its evidence, not a guarantee that no possible error remains.

The predecessor releases are pinned in the original [pilot receipt](publication.json) and [continuation receipt](publication-continuation-seed0-gen100.json). [Raw pilot evidence](https://huggingface.co/datasets/BurnyCoder/novelty-es-pilot-artifacts/tree/949e2dcb385bc2b7088c40444ef879b76596317d) and [raw continuation evidence](https://huggingface.co/datasets/BurnyCoder/novelty-es-continuation-seed0-gen100-artifacts/tree/64ca85166e198f2e064f51b70f2228228d65e814) remain unchanged. New documentation revisions are recorded separately in the [correction publication receipt](publication-corrections-2026-11-12.json).

## Confirmed errors and corrections

| Problem | Correction | Impact on historical findings |
|---|---|---|
| Tables placed fresh greedy accuracy beside unlabeled sampled pass@k | Label every sampled column as original-test, including the continuation; label hard as the five-number split and probes as training examples | Existing scores unchanged; prevents comparing different splits as the same endpoint |
| Reused original-model outcomes appeared in continuation evaluation totals without sufficient distinction | Separate stored outcomes, new response events, parent fresh evaluations, cumulative training and added training | Token measurements unchanged; stored totals are not newly consumed compute |
| Restart-adjusted phase durations were called compute time | Label recorded phase wall intervals, including overhead and excluding offline time and unknown unlogged outage tail | No GPU compute-hour claim is supported |
| Complete response strings could be read as complete-token embedding coverage | Document the pinned MiniLM cap and measure existing probe token lengths | No observed probe truncation in the saved records; no historical embedding policy changed |
| Arithmetic description omitted actual parser syntax | Document restricted Python syntax, bare expression or a single answer tag, surrounding-prose handling and rejected syntax | Independent grading agrees with saved grades; historical scores preserved |
| Uniform sampling docstring overstated the accepted data distribution | Distinguish uniform operand proposals from rejection-biased witnessed problems | No dataset changed; generalization remains distribution-specific |
| Original initialization and complete recovery-state preservation sounded freshly proven | Distinguish historical audit assertions from surviving tensor checks | Missing initial snapshots limit independent reproducibility; no invented equality result |
| EVOL-RL was characterized only as transductive | State that its paper reports both same-prompt adaptation and held-out transfer; identify its semantic embedding setup | Related-work interpretation corrected, no project score affected |
| Partial sampled groups were not checked for repeated/out-of-range sample indices | Reject malformed partial groups; independent verification checks exact identity/order/coverage too | Existing complete evaluations pass; reporting now fails explicitly on these corrupt inputs |
| Regenerating the pilot could erase the continuation from the aggregate | Maintain a per-study summary registry and regenerate both with `report-all` | Both studies remain visible regardless of regeneration order |
| Recovery commands pointed at an expired execution and restoration omitted separate exports | Provide pinned evidence-plus-model restoration and read-only verify/report commands | No expired training attempt restarted |
| Reporting wrote into evidence directories and could embed an old summary in a new snapshot | Add separate staged output, overlap checks, source-hash checks and regenerated snapshot summaries | Raw logs, weights, frozen registrations, deadlines and original receipts preserved |
| A log-close docstring implied power-loss durability; an unused JSONL helper silently accepted malformed final lines | Narrow the flush description and remove the unused helper | No active historical experimental path changed |

The detailed method reconciliation links the [ExTra paper and pinned caller/advantage code](../docs/literature.md#extra-implementation-reconciliation), the [ES at Scale numerical path and license](../docs/literature.md#es-at-scale-source-reuse-scope), and the [pass@k derivation](https://arxiv.org/html/2107.03374). Central papers were read in full, including appendices. Peripheral references were checked against their primary method descriptions for the narrow statements in the literature table; their performance claims were not reproduced.

## Independently regenerated findings

The following block is generated by `report-all` from the independent JSON evidence. It must not be edited as hand-entered results.

<!-- BEGIN INDEPENDENT VERIFIED RESULTS -->
| Study / primary split | Main correct / problems | Control correct / problems | Difference (pp) | Paired 95% interval (pp) | Main candidate weights changed |
| --- | --- | --- | --- | --- | --- |
| pilot / test | 23 / 2048 | 23 / 2048 | 0.000 | [-0.244, 0.244] | 14 |
| continuation / fresh | 151 / 2048 | 158 / 2048 | -0.342 | [-1.025, 0.342] | 182 |

Intervals resample problems conditional on the realized training seed. Changed-weight counts are cumulative recorded populations, not additional continuation-only counts.

| Log | Probe response events | Above embedding cap | Maximum input tokens | Cap including special tokens |
| --- | --- | --- | --- | --- |
| pilot | 18240 | 0 | 161 | 256 |
| continuation | 77760 | 0 | 143 | 256 |

Probe-event counts include repeated logged responses; the continuation log excludes its pilot prefix. Sources: [independent raw-record checks](independent-verification-2026-11-12.json) and [pinned-tokenizer measurements](embedding-inputs-2026-11-12.json).
<!-- END INDEPENDENT VERIFIED RESULTS -->

The primary conclusion is unchanged: neither comparison demonstrates a novelty advantage for the realized training seed. The main arm did receive nonzero novelty-induced coefficient changes. That local diagnostic does not replay a counterfactual optimization trajectory. Both trained checkpoints improve over the original checkpoint on their named primary sets; the longer run also improves over its parent checkpoints on the same fresh problems. These descriptive findings do not establish seed robustness, general language improvements, or superiority on other data distributions. See the complete [pilot](experiment-pilot.md) and [continuation](experiment-continuation-seed0-gen100.md) tables, including negative and harder-set outcomes.

## Verification and impact assessment

The [independent verifier](../src/novelty_es/verification.py) uses a separate precedence parser with exact rational arithmetic, while Python parsing enforces the historical syntax boundary. It checks all witnesses and split hashes; exact evaluation coverage and sampling identities; raw response grading and token lengths; training prompt order, batch offsets and perturbation seeds; committed quality/accuracy/validity/token totals; recorded update coefficients; export tensors versus FP32 master casts; continuation metric prefixes; and reused baseline rows. It hashes evidence before and after. Only documented live dashboard bookkeeping and download caches are excluded from the stable-evidence inventory. The [saved results](independent-verification-2026-11-12.json) contain file-level hashes and explicit evidence gaps.

The [presentation checker](../scripts/verify_presentations.py) independently verifies individual and paired intervals, baseline comparisons, fresh parent changes, difference-in-differences, pass@k, expression diversity, example selection, resource counts and website snapshot correspondence. Its [result](independent-presentations-2026-11-12.json) also compares server-side GitHub merge times with persisted execution clocks, verifies the copied interrupted evidence against frozen hashes, directly compares the recovered main tensors, and rechecks the failed-write response replay. [Tokenization-only checks](../scripts/check_embedding_inputs.py) use the pinned tokenizer and existing strings, without constructing or executing an embedding model.

No discovered grading, coverage or coefficient discrepancy changes the primary estimates. The missing historical embedding/archive trajectory prevents an independent reconstruction of every novelty distance. No new inference or optimization was used to fill that gap. Registered endpoints, historical accepted syntax and weighting remain unchanged. The continuation was chosen after observing the pilot, and the recovery followed an expired attempt; the revision retains that timing rather than treating either as an untouched initial preregistration.

Focused regression checks cover malformed sampled records, split labels, restoration destinations, overlap rejection, staged byte preservation, aggregate preservation, stale snapshot avoidance and portable Hub evidence links. The locked test suite also exercises the existing synthetic stand-in pipeline, fault recovery and analytic vector/tensor behavior. Those fixtures are software checks, not new research-model experiments; the [test record](independent-test-results-2026-11-12.json) contains the observed outcomes. [Citation checks](independent-citations-2026-11-12.json) distinguish an unavailable URL from evidence that a claim is false; URL reachability alone does not validate a source's contents.

Both corrected papers were rendered and inspected page by page; the [PDF inspection record](independent-pdf-review-2026-11-12.json) identifies the reviewed bytes. The public HTTPS dashboard's metric, range and matched-example controls were exercised successfully. Earlier failures of the detached Chrome interface and local-file URL policy were resolved by selecting the already-created public HTTPS tab. The [UI inspection record](independent-ui-2026-11-12.json) distinguishes the predecessor inspection from checks of the corrected deployment. The [restoration record](independent-restoration-2026-11-12.json) documents the real download and comparison against the original immutable releases.

## Unresolved evidence and inspection limits

- Generation-zero tensors and the initial recovered control checkpoint were overwritten. Old equality flags do not substitute for them. Final exports and the unchanged recovered main are directly checkable.
- A complete per-generation embedding/archive trajectory was not retained. Raw recorded novelty can be reconciled with applied coefficients, but all historical distances cannot be recomputed without additional model inference.
- The exact outage instant, the unlogged tail before it, and isolated GPU compute time are unavailable. Local event clocks establish recorded ordering, not independently attested clock accuracy.
- Internal split separation does not exclude pretraining contamination. A single paired training seed does not establish robustness across seeds, and expression-tree diversity is not complete semantic or algebraic equivalence.
- Historical statements about who held the Windows file lock, exact manual activity, or which background task caused timing differences cannot be proven from the saved logs. They remain explicitly uncertain or historical.

## Reproduction and revisions

Use the [restoration workflow](../docs/continuation.md) to obtain immutable evidence and separate model exports. `novelty-es verify` requires explicit pilot, continuation, data, fresh-data and output paths. `novelty-es report-all` regenerates both releases, their aggregate, figures, papers, cards and static website into a separate tree. The auxiliary presentation and tokenizer commands above analyze only existing evidence. Run from the repository root with the locked uv environment.

Documentation-only Hub commits retain the predecessor revisions and unchanged weight files. The correction receipt records remote revisions, changed-file hashes and unchanged-file checks; GitHub changes use a merged reviewable pull request and the existing Pages workflow. Old registrations and receipts are retained so the before/after record remains inspectable.
