# Independent scientific audit of the novelty ES results

Libor “Burny” Burian · 2026-11-26

## Scientific verdict

The observed novelty advantage predicted by H1 failed on both registered sets. The paired intervals include zero. The uncertainty concerns new problems for these checkpoints. Training-seed variability, equivalence and the effect of retraining with a stricter scorer are unidentified. The expired continuation had no paired endpoint within its allowance. The later recovered comparison uses the separately registered clock and reused main checkpoint.

| Study / primary set | Novelty correct | Control correct | Both / novelty only / control only / neither | Difference (pp) | Paired 95% interval (pp) | Exact two-sided p, post hoc |
|---|---:|---:|---|---:|---|---:|
| pilot / test | 23/2048 | 23/2048 | 20 / 3 / 3 / 2022 | +0.000 | [-0.244, +0.244] | 1 |
| continuation / fresh | 151/2048 | 158/2048 | 130 / 21 / 28 / 1869 | -0.342 | [-1.025, +0.342] | 0.391603 |

Each bootstrap draw resamples whole problem IDs. The exact sensitivity test conditions on the discordant count and evaluates a two-sided fair-binomial tail. The original, harder, parent, probe and sampled measures are descriptive secondary outcomes. The generator fixes the proportion of three- and four-number problems and excludes used multisets. The registered unstratified percentile procedure is an approximation to repeated problem sampling. A sparse or zero-width interval cannot establish equivalence. [Registered procedure](../preregistration.md) · [Matched-pair inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC9447366/).

## Training reward discrepancy

The frozen text permits one answer block or a bare expression and rejects multiple answer tags. The historical regex counts complete matches and can ignore a stray opening or closing tag. The new reader counts both literal delimiters, checks their order and then evaluates the permitted Python arithmetic with exact fractions. Prose outside a single balanced block remains permitted. This is a retrospective reading of the written rule.

| Training stage | Arm | Unique committed responses affected | Of which full credit | Probe eligibility changes | Affected updates | Changed coefficients on the same recorded populations |
|---|---|---:|---:|---:|---:|---:|
| pilot | main | 9 | 0 | 0 | 6 | 180 |
| pilot | reward_only | 11 | 0 | 0 | 7 | 210 |
| continuation | main | 39 | 0 | 0 | 20 | 600 |
| continuation | reward_only | 37 | 1 | 1 | 21 | 630 |

Continuation training counts cover added generations only. Each row-level change has a timestamp, generation, candidate, problem, old/new reward, old/new candidate mean and coefficient in the machine-readable evidence. Changing one reward can change other candidates’ coefficients through population standardization. The local coefficient calculation holds the recorded population and novelty fixed. A changed gate on the control cannot add a bonus because its coefficient is zero. Subsequent outputs and checkpoints under stricter training are unobserved.

Full-credit example in continuation, reward_only, update 80, candidate 20, problem `train-03770`. The opening/closing delimiter counts are 1/2. The stored candidate mean reward was 0.125000; regrading the same responses gives 0.104167. Its recorded update coefficient was +0.071796. On the same population the stricter score gives -1.589323.

## Evaluation and provenance

| Study | Arm | Split | Responses | Correct, historical / balanced | Legal, historical / balanced |
|---|---|---|---:|---|---|
| pilot | original | test | 2048 | 6 / 6 | 600 / 600 |
| pilot | original | hard | 1024 | 1 / 1 | 109 / 109 |
| pilot | original | probes | 16 | 0 / 0 | 4 / 4 |
| pilot | original | sampled | 4096 | 5 / 5 | 304 / 303 |
| pilot | main | test | 2048 | 23 / 23 | 1559 / 1559 |
| pilot | main | hard | 1024 | 0 / 0 | 21 / 21 |
| pilot | main | probes | 16 | 0 / 0 | 14 / 14 |
| pilot | main | sampled | 4096 | 11 / 11 | 786 / 782 |
| pilot | reward_only | test | 2048 | 23 / 23 | 1634 / 1634 |
| pilot | reward_only | hard | 1024 | 1 / 1 | 36 / 36 |
| pilot | reward_only | probes | 16 | 0 / 0 | 13 / 13 |
| pilot | reward_only | sampled | 4096 | 9 / 9 | 710 / 709 |
| continuation | original | test | 2048 | 6 / 6 | 600 / 600 |
| continuation | original | hard | 1024 | 1 / 1 | 109 / 109 |
| continuation | original | probes | 16 | 0 / 0 | 4 / 4 |
| continuation | original | sampled | 4096 | 5 / 5 | 304 / 303 |
| continuation | original | fresh | 2048 | 5 / 5 | 609 / 609 |
| continuation | main | fresh | 2048 | 151 / 151 | 1961 / 1961 |
| continuation | main | test | 2048 | 135 / 135 | 1965 / 1965 |
| continuation | main | hard | 1024 | 4 / 4 | 743 / 743 |
| continuation | main | probes | 16 | 3 / 3 | 15 / 15 |
| continuation | main | sampled | 4096 | 93 / 93 | 1760 / 1759 |
| continuation | reward_only | fresh | 2048 | 158 / 158 | 1963 / 1963 |
| continuation | reward_only | test | 2048 | 147 / 147 | 1976 / 1976 |
| continuation | reward_only | hard | 1024 | 9 / 9 | 783 / 783 |
| continuation | reward_only | probes | 16 | 3 / 3 | 16 / 16 |
| continuation | reward_only | sampled | 4096 | 101 / 101 | 1885 / 1882 |
| continuation | parent_main | fresh | 2048 | 24 / 24 | 1610 / 1610 |
| continuation | parent_reward_only | fresh | 2048 | 26 / 26 | 1694 / 1694 |

The sampled rows use the original-test subset in both studies. pass@k estimates the success probability of a uniform k-subset of saved samples. The interpretation for independent model draws assumes independent sampling. A greedy correct-tree count equals the number of successes. Canonical trees from multiple samples measure syntactic variation within problems. The training-probe and harder-distribution results have distinct denominators. [Chen et al., Eq. 1](https://arxiv.org/pdf/2107.03374).


Each evaluation or validation discrepancy changes the recorded descriptive reward or legality flag for that response. These responses have no candidate fitness, archive entry or novelty eligibility. Evaluation tables count reused baseline outcomes in each release. The evidence identifies them for cross-study deduplication.

### Change on the same fresh problems

| Secondary comparison | Difference (pp) | Paired 95% interval (pp) |
|---|---:|---|
| Novelty final minus parent | +6.201 | [+4.932, +7.422] |
| Control final minus parent | +6.445 | [+5.176, +7.715] |
| Novelty change minus control change | -0.244 | [-0.977, +0.488] |

The last row first combines all four Boolean outcomes within each problem and then resamples problems. Both arms improved on these problems with further training. The interval for their difference in improvement includes zero. These secondary contrasts are unadjusted for multiplicity.

- **pilot**. Checked 77,040 rendered prompts, 77,024 token-to-text decodes, 1,347 committed evaluation batches and 8 validation populations. Matched 54,720 committed training rows. Additional identical evaluation batches: 1. Uncommitted or incomplete evaluation attempts: 1.
  Before applying the first differing coefficient vector (update 2), 2,880 paired response comparisons had 0 differences in the checked fields. Identical behavior on these prompts does not identify all initial tensor values.
- **continuation**. Checked 260,096 rendered prompts, 260,080 token-to-text decodes, 1,538 committed evaluation batches and 34 validation populations. Matched 233,280 committed training rows. Additional identical evaluation batches: 0. Uncommitted or incomplete evaluation attempts: 0.

## Method and evidence limits

Population-standardized task quality weights Gaussian perturbations. The main arm adds unnormalized correctness-gated nearest-neighbor cosine novelty on fixed training probes. Per-tensor noise streams, candidate siblings, frozen training memory and the omission of prefix regeneration are project adaptations. Embedding distance is a proxy for response variation; it does not identify mathematical reasoning modes. The tokenizer truncates before embedding. [ES at Scale, Algorithm 2](https://arxiv.org/html/2509.24372v3#S3) · [ExTra, Eqs. 1–2 and Appendix B](https://arxiv.org/html/2606.24994v1).

The shared generation configuration applies its inherited repetition penalty during greedy and sampled decoding. Greedy selects the highest processed token score. Sampling uses all-token top-p and disables top-k. [Pinned Qwen defaults](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/generation_config.json) · [Generation options](https://huggingface.co/docs/transformers/main_classes/text_generation).

The available final masters, exports and archive tensors can be inspected. Generation-zero tensors, the initial recovered control checkpoint and a complete per-generation embedding history are unavailable. Full initialization equality, every historical novelty distance and the counterfactual trajectory cannot be independently established. Raw prompt checks constrain logged exposure; pretraining overlap and unlogged human exposure remain unknown. One paired seed and a pilot-motivated continuation do not provide independent training replications. Resource counts distinguish reused outcomes, committed training tokens, retries and wall intervals.

[Row-level evidence and secondary contrasts](registered-rule-evidence-2026-11-26.json) · [Claim-by-claim evidence map](claim-map-2026-11-26.md) · [Server chronology and original object hashes](provenance-2026-11-26.json) · [Independent historical-score checks](historical-verification-2026-11-26.json) · [Embedding input measurements](embedding-inputs-2026-11-26.json).

[Pilot raw evidence](https://huggingface.co/datasets/BurnyCoder/novelty-es-pilot-artifacts/tree/949e2dcb385bc2b7088c40444ef879b76596317d) · [Recovery raw evidence](https://huggingface.co/datasets/BurnyCoder/novelty-es-continuation-seed0-gen100-artifacts/tree/64ca85166e198f2e064f51b70f2228228d65e814).

## Independently checkable novelty calculations

| Stage | Arm | Incorrect gates | Correct, empty pool | Correct, all distance inputs retained | Correct, missing distance inputs | Complete candidate means / with a correct probe | Maximum absolute error in checked means |
|---|---|---:|---:|---:|---:|---|---:|
| pilot | main | 9049 | 2 | 4 | 65 | 514 / 6 | 0 |
| added | main | 35094 | 4 | 8 | 3774 | 431 / 2 | 0 |
| pilot | reward_only | 9057 | 1 | 3 | 59 | 520 / 4 | 0 |
| added | reward_only | 35105 | 5 | 3 | 3767 | 406 / 3 | 0 |

The historical Boolean gate determines each probe's eligibility in this reconstruction. A correct probe with an empty reference pool receives the registered default. A stored cosine distance can be recalculated when the query and every required sibling and archive embedding survive. The candidate mean uses all fixed probes. The archive record lists each retained snapshot and the replayed insertion RNG checks. [Saved-tensor evidence](archive-2026-11-26.json).

- Pilot had 18,240 logged probe responses. The maximum tokenized length was 161 and 0 exceeded the encoder limit of 256. [Tokenizer measurements](embedding-inputs-2026-11-26.json).
- Continuation had 77,760 logged probe responses. The maximum tokenized length was 143 and 0 exceeded the encoder limit of 256. [Tokenizer measurements](embedding-inputs-2026-11-26.json).
