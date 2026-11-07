# Literature and reuse audit

Reviewed for this project on 2026-11-07. The first experiment is an adaptation of an LLM novelty objective to full-weight evolution strategies (ES). Published findings below belong to their authors; none are results of this project. The review distinguishes changing model parameters, changing prompts, and evolving generated artifacts.

## Closest parameter-training methods

| Work | What changes / novelty mechanism | Relevance and reuse decision |
|---|---|---|
| [ExTra](https://arxiv.org/abs/2606.24994), [full text](https://arxiv.org/html/2606.24994v1), [code](https://github.com/allen4747/extra) | GRPO model training; correct-response embedding distance to historical and sibling responses; separate prefix regeneration | Closest lightweight historical LLM novelty component. Adapt its memory under Apache-2.0; omit regeneration and GRPO. |
| [ES at Scale](https://arxiv.org/abs/2509.24372), [full text](https://arxiv.org/html/2509.24372v3), [code](https://github.com/VsonicV/es-at-scale) | Full-weight Gaussian perturbations and reward-weighted updates, including Countdown | Optimization reference. Its Academic Public License permits this project's noncommercial research scope, subject to its redistribution terms. The running pilot uses an independent implementation of the registered equations. |
| [EVOL-RL](https://arxiv.org/abs/2509.15194), [v4](https://arxiv.org/html/2509.15194v4), [code](https://github.com/YujunZhou/EVOL-RL) | GRPO with majority-derived pseudo-label rewards and within-group semantic novelty | Direct LLM novelty training, but no historical archive. Its heavier semantic scorer and transductive setting do not match this pilot. |
| [DARLING](https://arxiv.org/abs/2509.02534), [code](https://github.com/facebookresearch/darling) | Joint quality/diversity RL; semantic partitions distinguish response families | MIT code offers useful diversity evaluation ideas. Reported n-gram reward failure motivates an independent functional diagnostic. |
| [MoDA](https://arxiv.org/abs/2609.14896), [code](https://github.com/yuanjiayiy/mode-conditioned-diversity-alignment) | Mode-conditioned RL with quality-gated nearest-neighbor response diversity | Apache-2.0 implementation; relevant gate and diversity concepts, but a substantially different policy/compute setup. |
| [AC/DC](https://arxiv.org/abs/2604.14969), [full text](https://arxiv.org/html/2604.14969v1) | Coevolution of task archives and LLM experts; model merging, singular-value mutation, dominated novelty | Closest broader evolutionary LLM quality-diversity system. Requires compatible seed experts and task-generation machinery. No reusable official implementation verified in this review. |
| [Beyond the Best Guess](https://arxiv.org/abs/2608.12679) | Full-weight ES and solution coverage, without an explicit novelty reward | Motivates a reward-only ES control: ES itself can affect diversity. Finite sampling cannot establish that an answer has zero model probability. |

## ExTra implementation reconciliation

The selected upstream commit is `ea53e68036356f8f276e261e886fbbc102633be3`. Read the complete paper and the [memory implementation](https://github.com/allen4747/extra/blob/ea53e68036356f8f276e261e886fbbc102633be3/verl/recipe/extra/curiosity_memory.py), caller, and advantage code. The released embedding-memory function inserts all supplied embeddings; it does not itself enforce the paper's correct-answer memory. Our adapted `CorrectMemory` therefore accepts a Boolean exact-correctness mask for scoring and insertion. A positive legality reward is insufficient. We also separate scoring from insertion so history remains fixed across a generation. Attribution and the upstream license are retained in [NOTICE](../NOTICE).

Our sibling pool comprises perturbed models answering the same probe. ExTra uses rollout siblings. Its normalization, warmup, coefficient, and regeneration choices are not reproduced here. Model weights, task, data, rank shaping, and hardware also differ. The combination is an experiment, not a replication or a claim that either publication predicts our outcome.

## ES at Scale source-reuse scope

The project owner clarified on 2026-11-07 that this is strictly research with no commercial use. Rechecked the complete [Academic Public License at upstream commit `574a9d134da1ffce2a8bb812019899e5c96b588a`](https://github.com/VsonicV/es-at-scale/blob/574a9d134da1ffce2a8bb812019899e5c96b588a/LICENSE.txt): section 1 permits noncommercial purposes, including personal nonprofit use. The license is therefore not a blanket obstacle to source reuse for this project. The earlier wording implying that it prevented reuse was too broad.

Sections 2 and 3 require retaining the license and notices, identifying modifications and their dates, and distributing covered derivative works under the Academic Public License. If upstream code is incorporated later, its provenance and applicable license must be preserved; it cannot simply be relabeled as Apache-2.0. The current pilot contains no copied ES at Scale source, so this clarification changes documentation rather than the executed optimizer or its license. Its registered rank shaping and deterministic parameter reconstruction remain fixed during the paired run. Future research can reuse suitable upstream components under their actual terms.

The source audit at that commit read the complete [reward shaping](https://github.com/VsonicV/es-at-scale/blob/574a9d134da1ffce2a8bb812019899e5c96b588a/es_at_scale/utils/reward_shaping.py), [worker extension](https://github.com/VsonicV/es-at-scale/blob/574a9d134da1ffce2a8bb812019899e5c96b588a/es_at_scale/utils/worker_extension.py), and [trainer](https://github.com/VsonicV/es-at-scale/blob/574a9d134da1ffce2a8bb812019899e5c96b588a/es_at_scale/trainer/es_trainer.py). The reusable systems components include persistent vLLM engines, Ray population dispatch and inter-engine weight broadcast. Their numerical path uses z-score shaping, native-precision perturbations and perturb/subtract restoration. The pilot instead specifies average centered ranks and fresh candidates reconstructed from FP32 masters, with distinct parameter-name noise streams. Adopting the upstream trainer unchanged would therefore change the registered experiment. Any later integration must retain the chosen numerical semantics or be recorded as a methodological amendment before its results exist.

## Other nearby work and boundaries

| Work | Classification | Transferable insight |
|---|---|---|
| [Conti et al., novelty-seeking ES](https://arxiv.org/abs/1712.06560), [MIT source](https://github.com/uber-research/deep-neuroevolution) | Behavioral novelty + ES in control tasks | Archive-based exploration and reward/novelty controls; these results are not LLM evidence. |
| [Novelty-based ES and Decision Transformers](https://arxiv.org/abs/2502.06301), [repository](https://github.com/Mafi412/Novelty-based-Evolution-Strategies-and-Decision-Transformers) | Transformer weight evolution for control | Relevant architecture precedent, not language training. No code imported; license not verified. |
| [QDAIF](https://arxiv.org/abs/2310.13032), [OpenELM](https://github.com/CarperAI/OpenELM) | Primarily quality-diversity evolution of text artifacts; appendix also fine-tunes on archive data | MIT framework. Archive-to-fine-tuning is a second route; reported fine-tuning effects are mixed, not a guaranteed improvement. |
| [QD-LLM](https://arxiv.org/abs/2605.09781) | Prompt-embedding evolution with a frozen LLM, plus archive-based fine-tuning experiments | Does not satisfy the chosen full-weight ES method. No verified reusable source imported. |
| [TileGPT](https://arxiv.org/abs/2405.09997) | QD-generated design data followed by language-model training | Diversity can enter through training data instead of the optimizer. |
| [MarioGPT](https://papers.neurips.cc/paper_files/paper/2023/file/a9bbeb2858dfbdbd4c19814e5d80ec60-Paper-Conference.pdf) | Language-model level generation plus novelty search over levels | Generated-artifact search is distinct from directly evolving LLM weights. |
| [DivPO](https://arxiv.org/abs/2501.18101) | Diversity-aware preference pair selection | Another parameter-training approach, not archive-based ES. |
| [Diversity-Quality Optimization](https://arxiv.org/abs/2509.04784), [code](https://github.com/fairytale9/diversity-quality-optimization) | Group diversity/quality optimization | Determinantal group rewards are a possible later alternative; no code imported or license verified. |
| [VPO](https://arxiv.org/abs/2605.22817) | Group/vector reward optimization | Alternative diversity training, not the selected historical nearest-neighbor objective. |
| [Curiosity-Driven Exploration](https://arxiv.org/abs/2509.09675) | Curiosity bonuses for LLM RL | Broader intrinsic-reward context; abstract/method overview only in this review. |
| [OCNR](https://proceedings.mlr.press/v306/lee26ay.html) | Novelty in self-generated training tasks | Proceedings abstract inspected; full paper was not accessible. Not an implementation dependency. |
| [NoveltyBench](https://arxiv.org/abs/2504.05228), [code](https://github.com/novelty-bench/novelty-bench) | Evaluation of distinct, useful generations | Keep output quality and meaningful diversity separate; not a training algorithm. |
| [EGGROLL](https://arxiv.org/abs/2511.16652) | Efficient low-rank perturbations with aggregate full-weight updates | Possible later systems optimization; not synonymous with LoRA. Initial implementation avoids changing the Gaussian-noise specification. |
| [MeZO](https://arxiv.org/abs/2305.17333) | Zeroth-order LLM optimization | Memory-efficiency context; no explicit novelty archive. |

## Models, datasets, and research practice

The base [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [MiniLM card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) identify Apache-2.0 models; exact revisions are in the experiment configuration. Countdown data here are generated independently and come with symbolic witnesses. This reduces direct benchmark contamination but cannot exclude prior exposure to similar arithmetic patterns.

The deferred scratch phase uses [TinyStories](https://arxiv.org/abs/2305.07759); its evidence about small language models is not evidence that ES can train them competitively. Dataset licensing must be recorded when that phase is frozen.

The protocol follows the concerns in [Lones, v3](https://arxiv.org/abs/2108.02497v3), [Lipton and Steinhardt](https://arxiv.org/abs/1807.03341), and the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist): preserve held-out data, state the scope of claims, distinguish method changes from reproductions, expose failed experiments, and describe uncertainty. One training seed permits a conditional paired problem comparison, not a population claim across training runs. Search coverage is a focused research audit, not a systematic-review completeness claim.
