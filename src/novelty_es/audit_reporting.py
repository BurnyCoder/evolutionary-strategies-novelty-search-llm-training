"""Publish the dated independent audit from its measured records, preserving historical scores."""

import shutil
from pathlib import Path

from .verification import read, sha256

DATE = "2026-11-26"
FOLDER = Path("reports/reviews")
EVIDENCE = FOLDER / f"registered-rule-evidence-{DATE}.json"
REVIEW = FOLDER / f"scientific-audit-{DATE}.md"
GITHUB = "https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training"
REVIEW_URL = f"{GITHUB}/blob/main/{REVIEW.as_posix()}"
QWEN = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/generation_config.json"


def summary_for(evidence, directory, study):
    """Bind the new interpretation to the precise raw rows independently analyzed."""
    data = evidence["studies"][study]
    for path, digest in data["input_hashes"].items():
        if sha256(directory / path) != digest:
            raise ValueError(f"Audit evidence does not describe the supplied {study} input {path}")
    training = data["trace"]["training_discrepancies"]
    evaluations = data["evaluations"]
    split = data["primary_split"]
    return dict(
        date=evidence["revision_date"],
        review=REVIEW_URL,
        experiment_sha256=data["input_hashes"]["experiment.json"],
        malformed_training={arm: sum(x["condition"] == arm for x in training) for arm in ("main", "reward_only")},
        changed_probe_gates=sum(x["novelty_eligibility_changed"] for x in training),
        populations=len(data["trace"]["affected_populations"]),
        strict_primary=data["comparisons"]["primary"],
        repetition_penalty=data["trace"]["decoding"]["repetition_penalty"],
        primary_correct_changes={
            a: evaluations[a][split]["correct"] - evaluations[a][split]["historical_correct"]
            for a in ("main", "reward_only")
        },
    )


def applicable_summary(directory, override=None):
    """Attach the published correction to individual reports and dashboards for these exact runs."""
    manifest = directory / "experiment.json"
    source = override or EVIDENCE
    if source.is_file() and manifest.is_file():
        evidence = read(source)
        digest = sha256(manifest)
        for name, study in evidence["studies"].items():
            if study["input_hashes"]["experiment.json"] == digest:
                return summary_for(evidence, directory, name)
    if override is not None:
        raise ValueError("The audit evidence must identify the supplied experiment")
    return None


def audit_description(item):
    """Shared prose for the papers, site, overview, reports and Hub cards."""
    p = item["strict_primary"]
    return (
        f"The historical scorer accepted one complete answer block even when an extra delimiter appeared. "
        f"Applying the balanced-tag reading to the saved training responses changes {item['malformed_training']['main']} "
        f"novelty-arm and {item['malformed_training']['reward_only']} control-arm rewards in this training stage, "
        f"including {item['changed_probe_gates']} changes to fixed-probe novelty eligibility. "
        f"On the saved final outputs the balanced-tag difference is {100 * p['difference']:+.3f} percentage points "
        f"(paired 95% interval {100 * p['low']:+.3f} to {100 * p['high']:+.3f}). "
        f"The novelty checkpoint alone solved {p['main_only']} problems and the control alone solved {p['control_only']}. "
        f"The post-hoc two-sided exact McNemar p-value is {p['exact_two_sided_p']:.4g}. "
        f"Both decoding paths inherit a repetition penalty of {item['repetition_penalty']:g}. "
        "The measured checkpoints reflect the historical rewards. The outcomes of training with balanced tags are unknown."
    )


def write_review(path, output):
    """Materialize the audit and its evidence alongside both generated releases."""
    evidence = read(path)
    target = output / REVIEW
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != (output / EVIDENCE).resolve():
        shutil.copyfile(path, output / EVIDENCE)
    rows = [
        "# Independent scientific audit of the novelty ES results",
        "",
        f"Libor “Burny” Burian · {evidence['revision_date']}",
        "",
        "## Scientific verdict",
        "",
        "The observed novelty advantage predicted by H1 failed on both registered sets. The paired intervals include zero. "
        "The uncertainty concerns new problems for these checkpoints. Training-seed variability, equivalence and the effect "
        "of retraining with a stricter scorer are unidentified. The expired continuation had no paired endpoint within its "
        "allowance. The later recovered comparison uses the separately registered clock and reused main checkpoint.",
        "",
        "| Study / primary set | Novelty correct | Control correct | Both / novelty only / control only / neither | Difference (pp) | Paired 95% interval (pp) | Exact two-sided p, post hoc |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for name, s in evidence["studies"].items():
        p = s["comparisons"]["primary"]
        counts = [s["evaluations"][a][s["primary_split"]]["correct"] for a in ("main", "reward_only")]
        joint = " / ".join(str(p[k]) for k in ("both_correct", "main_only", "control_only", "both_wrong"))
        rows.append(
            f"| {name} / {s['primary_split']} | {counts[0]}/{p['n']} | {counts[1]}/{p['n']} | {joint} | "
            f"{100 * p['difference']:+.3f} | [{100 * p['low']:+.3f}, {100 * p['high']:+.3f}] | {p['exact_two_sided_p']:.6g} |"
        )
    rows.extend(
        [
            "",
            "Each bootstrap draw resamples whole problem IDs. The exact sensitivity test conditions on the discordant "
            "count and evaluates a two-sided fair-binomial tail. The original, harder, parent, probe and sampled measures "
            "are descriptive secondary outcomes. The generator fixes the proportion of three- and four-number problems and "
            "excludes used multisets. The registered unstratified percentile procedure is an approximation to repeated problem sampling. "
            "A sparse or zero-width interval cannot establish equivalence. "
            "[Registered procedure](../preregistration.md) · [Matched-pair inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC9447366/).",
            "",
            "## Training reward discrepancy",
            "",
            "The frozen text permits one answer block or a bare expression and rejects multiple answer tags. "
            "The historical regex counts complete matches and can ignore a stray opening or closing tag. The new reader counts "
            "both literal delimiters, checks their order and then evaluates the permitted Python arithmetic with exact fractions. "
            "Prose outside a single balanced block remains permitted. This is a retrospective reading of the written rule.",
            "",
            "| Training stage | Arm | Unique committed responses affected | Of which full credit | Probe eligibility changes | Affected updates | Changed coefficients on the same recorded populations |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, s in evidence["studies"].items():
        trace = s["trace"]
        for arm in ("main", "reward_only"):
            altered = [x for x in trace["training_discrepancies"] if x["condition"] == arm]
            pops = [x for x in trace["affected_populations"] if x["condition"] == arm]
            rows.append(
                f"| {name} | {arm} | {len(altered)} | {sum(x['historical_correct'] for x in altered)} | "
                f"{sum(x['novelty_eligibility_changed'] for x in altered)} | {len(pops)} | "
                f"{sum(x['changed_weights_holding_recorded_novelty'] for x in pops)} |"
            )
    rows.extend(
        [
            "",
            "Continuation training counts cover added generations only. Each row-level change has a timestamp, generation, "
            "candidate, problem, old/new reward, old/new candidate mean and coefficient in the machine-readable evidence. "
            "Changing one reward can change other candidates’ coefficients through population standardization. The local coefficient "
            "calculation holds the recorded population and novelty fixed. A changed gate on the control cannot add a bonus because its "
            "coefficient is zero. Subsequent outputs and checkpoints under stricter training are unobserved.",
            "",
        ]
    )
    for name, s in evidence["studies"].items():
        for x in s["trace"]["training_discrepancies"]:
            if x["historical_correct"] != x["registered_correct"]:
                rows.append(
                    f"Full-credit example in {name}, {x['condition']}, update {x['update']}, candidate {x['candidate']}, "
                    f"problem `{x['problem_id']}`. The opening/closing delimiter counts are {x['opening_tags']}/{x['closing_tags']}. "
                    f"The stored candidate mean reward was {x['historical_candidate_quality']:.6f}; regrading the same responses gives "
                    f"{x['registered_candidate_quality']:.6f}. Its recorded update coefficient was {x['recorded_weight']:+.6f}. "
                    f"On the same population the stricter score gives {x['fixed_novelty_registered_weight']:+.6f}."
                )
    rows.extend(
        [
            "",
            "## Evaluation and provenance",
            "",
            "| Study | Arm | Split | Responses | Correct, historical / balanced | Legal, historical / balanced |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for name, s in evidence["studies"].items():
        for arm, splits in s["evaluations"].items():
            for split, r in splits.items():
                rows.append(
                    f"| {name} | {arm} | {split} | {r['responses']} | {r['historical_correct']} / {r['correct']} | "
                    f"{r['historical_valid']} / {r['valid']} |"
                )
    rows.extend(
        [
            "",
            "The sampled rows use the original-test subset in both studies. pass@k estimates the success probability of "
            "a uniform k-subset of saved samples. The interpretation for independent model draws assumes independent sampling. "
            "A greedy correct-tree count equals the number of successes. Canonical trees from multiple samples measure syntactic "
            "variation within problems. The training-probe and harder-distribution results have distinct denominators. "
            "[Chen et al., Eq. 1](https://arxiv.org/pdf/2107.03374).",
            "",
        ]
    )
    rows.extend(
        [
            "",
            "Each evaluation or validation discrepancy changes the recorded descriptive reward or legality flag for "
            "that response. These responses have no candidate fitness, archive entry or novelty eligibility. "
            "Evaluation tables count reused baseline outcomes in each release. The evidence identifies them for cross-study deduplication.",
            "",
        ]
    )
    longer = evidence["studies"]["continuation"]["comparisons"]
    rows.extend(
        [
            "### Change on the same fresh problems",
            "",
            "| Secondary comparison | Difference (pp) | Paired 95% interval (pp) |",
            "|---|---:|---|",
        ]
    )
    for key, label in (
        ("main_change", "Novelty final minus parent"),
        ("reward_only_change", "Control final minus parent"),
        ("difference_in_changes", "Novelty change minus control change"),
    ):
        contrast = longer[key]
        rows.append(
            f"| {label} | {100 * contrast['difference']:+.3f} | [{100 * contrast['low']:+.3f}, {100 * contrast['high']:+.3f}] |"
        )
    rows.extend(
        [
            "",
            "The last row first combines all four Boolean outcomes within each problem and then resamples problems. "
            "Both arms improved on these problems with further training. The interval for their difference in improvement "
            "includes zero. These secondary contrasts are unadjusted for multiplicity.",
            "",
        ]
    )
    for name, s in evidence["studies"].items():
        t = s["trace"]
        rows.append(
            f"- **{name}**. Checked {t['counts']['rendered_prompts']:,} rendered prompts, {t['counts']['decoded_responses']:,} "
            f"token-to-text decodes, {t['committed_evaluation_batches']:,} committed evaluation batches and "
            f"{t['validation_populations']} validation populations. Matched {t['unique_training_rows']:,} committed training rows. "
            f"Additional identical evaluation batches: {t['additional_identical_evaluation_batches']}. "
            f"Uncommitted or incomplete evaluation attempts: {t['counts'].get('uncommitted_or_incomplete_evaluation_attempts', 0)}."
        )
        if t["pre_update_paired_responses"]:
            rows.append(
                f"  Before applying the first differing coefficient vector (update {t['first_different_coefficient_update']}), "
                f"{t['pre_update_paired_responses']:,} paired response comparisons had {t['pre_update_different_responses']} "
                "differences in the checked fields. Identical behavior on these prompts does not identify all initial tensor values."
            )
    rows.extend(
        [
            "",
            "## Method and evidence limits",
            "",
            "Population-standardized task quality weights Gaussian perturbations. The main arm adds unnormalized correctness-gated "
            "nearest-neighbor cosine novelty on fixed training probes. Per-tensor noise streams, candidate siblings, frozen training "
            "memory and the omission of prefix regeneration are project adaptations. Embedding distance is a proxy for response "
            "variation; it does not identify mathematical reasoning modes. The tokenizer truncates before embedding. "
            "[ES at Scale, Algorithm 2](https://arxiv.org/html/2509.24372v3#S3) · [ExTra, Eqs. 1–2 and Appendix B](https://arxiv.org/html/2606.24994v1).",
            "",
            "The shared generation configuration applies its inherited repetition penalty during greedy and sampled decoding. "
            "Greedy selects the highest processed token score. Sampling uses all-token top-p and disables top-k. "
            f"[Pinned Qwen defaults]({QWEN}) · [Generation options](https://huggingface.co/docs/transformers/main_classes/text_generation).",
            "",
            "The available final masters, exports and archive tensors can be inspected. Generation-zero tensors, the initial recovered "
            "control checkpoint and a complete per-generation embedding history are unavailable. Full initialization equality, every historical "
            "novelty distance and the counterfactual trajectory cannot be independently established. Raw prompt checks constrain logged exposure; "
            "pretraining overlap and unlogged human exposure remain unknown. One paired seed and a pilot-motivated continuation do not provide "
            "independent training replications. Resource counts distinguish reused outcomes, committed training tokens, retries and wall intervals.",
            "",
            f"[Row-level evidence and secondary contrasts]({EVIDENCE.name}) · [Claim-by-claim evidence map](claim-map-{DATE}.md) · "
            f"[Server chronology and original object hashes](provenance-{DATE}.json) · [Independent historical-score checks](historical-verification-{DATE}.json) · "
            f"[Embedding input measurements](embedding-inputs-{DATE}.json).",
            "",
            "[Pilot raw evidence](https://huggingface.co/datasets/BurnyCoder/novelty-es-pilot-artifacts/tree/949e2dcb385bc2b7088c40444ef879b76596317d) · "
            "[Recovery raw evidence](https://huggingface.co/datasets/BurnyCoder/novelty-es-continuation-seed0-gen100-artifacts/tree/64ca85166e198f2e064f51b70f2228228d65e814).",
        ]
    )
    for stem in ("archive", "embedding-inputs"):
        filename = f"{stem}-{DATE}.json"
        candidates = [path.parent / filename, path.parent / f"{stem}.json", FOLDER / filename]
        companion = next((p for p in candidates if p.is_file()), None)
        if companion is None:
            raise FileNotFoundError(f"Generate the accompanying {stem} measurements before compiling the audit.")
        destination = target.parent / filename
        if companion.resolve() != destination.resolve():
            shutil.copyfile(companion, destination)
    archives = read(target.parent / f"archive-{DATE}.json")
    rows.extend(
        [
            "",
            "## Independently checkable novelty calculations",
            "",
            "| Stage | Arm | Incorrect gates | Correct, empty pool | Correct, all distance inputs retained | Correct, missing distance inputs | Complete candidate means / with a correct probe | Maximum absolute error in checked means |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for arm, result in archives["arms"].items():
        for stage, counts in result["stages"].items():
            rows.append(
                f"| {stage} | {arm} | {counts.get('incorrect_gate_zero', 0)} | {counts.get('correct_empty_pool', 0)} | "
                f"{counts.get('correct_with_tensor_distance', 0)} | {counts.get('correct_missing_distance_inputs', 0)} | "
                f"{counts.get('fully_checkable_candidate_means', 0)} / {counts.get('fully_checkable_with_correct_probe', 0)} | "
                f"{counts['max_abs_mean_error']:.3g} |"
            )
    rows.extend(
        [
            "",
            "The historical Boolean gate determines each probe's eligibility in this reconstruction. "
            "A correct probe with an empty reference pool receives the registered default. A stored cosine distance can be "
            "recalculated when the query and every required sibling and archive embedding survive. The candidate mean uses "
            "all fixed probes. The archive record lists each retained snapshot and the replayed insertion RNG checks. "
            f"[Saved-tensor evidence](archive-{DATE}.json).",
            "",
        ]
    )
    for stage, measured in read(target.parent / f"embedding-inputs-{DATE}.json").items():
        rows.append(
            f"- {stage.capitalize()} had {measured['logged_probe_responses']:,} logged probe responses. The maximum "
            f"tokenized length was {measured['max_tokens']} and {measured['above_limit']} exceeded the encoder limit "
            f"of {measured['limit']}. [Tokenizer measurements](embedding-inputs-{DATE}.json)."
        )
    target.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
