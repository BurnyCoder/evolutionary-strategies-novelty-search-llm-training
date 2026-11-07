"""Regenerate every result table/figure from persisted observations, never typed scores.

Matplotlib: https://matplotlib.org/stable/users/explain/quick_start.html .
ReportLab: https://docs.reportlab.com/reportlab/userguide/ch6_paragraphs/ .
This module can report partial experiments without presenting them as completed.
"""

from __future__ import annotations

import html
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .analysis import novelty_intervention, paired_bootstrap, pass_at_k
from .dashboard import dashboard_state, read_json
from .runtime import atomic_json


def logged_generation_work(path: Path) -> list[dict]:
    """Count final response events, including logged retries and abandoned batches.

    Stream the trace to avoid loading full prompts and partial texts into RAM.
    These are emitted-token counts, not padding or hardware-operation counts.
    """
    groups = defaultdict(lambda: {"responses": 0, "generated_tokens": 0})
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if line.endswith("\n"):
                        raise
                    break  # A running writer may have an unfinished final line.
                if event.get("event") == "response":
                    key = (event.get("condition", "calibration"), event.get("phase", "unknown"))
                    groups[key]["responses"] += 1
                    groups[key]["generated_tokens"] += event["generated_tokens"]
    return [{"condition": key[0], "phase": key[1], **values} for key, values in sorted(groups.items())]


def _split_summary(rows: list[dict], expected: int, replicates: int) -> dict:
    """Summarize observed rows and retain denominator/completeness separately."""
    if not rows:
        return {"n": 0, "expected": expected, "complete": False}
    ids = [row["problem_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate greedy evaluation problem IDs")
    accuracy = np.array([row["correct"] for row in rows], dtype=float)
    interval = paired_bootstrap(accuracy, np.zeros_like(accuracy), replicates)
    return {
        "n": len(rows),
        "expected": expected,
        "complete": len(rows) == expected,
        "accuracy": float(accuracy.mean()),
        "interval": [interval["low"], interval["high"]],
        "validity": float(np.mean([row["valid"] for row in rows])),
        "reward": float(np.mean([row["reward"] for row in rows])),
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "correct_expression_structures": len({(row["problem_id"], row["structure"]) for row in rows if row["correct"]}),
    }


def _paired_evaluations(first: dict, second: dict, replicates: int) -> dict:
    """Align exact test-problem identities before paired resampling."""
    a = {row["problem_id"]: row["correct"] for row in first["sets"]["test"]}
    b = {row["problem_id"]: row["correct"] for row in second["sets"]["test"]}
    if a.keys() != b.keys():
        raise ValueError("Paired test problem IDs do not align")
    ordered = sorted(a)
    return paired_bootstrap([a[key] for key in ordered], [b[key] for key in ordered], replicates)


def summarize(directory: Path) -> dict:
    """Build a machine-readable audit and registered comparison from raw evaluations."""
    experiment = read_json(directory / "experiment.json", {})
    config = experiment.get("config", read_json(Path("configs/pilot.json"), {}))
    manifest = experiment.get("dataset_manifest", read_json(Path("data/manifest.json"), {}))
    summary = {
        "methods": {},
        "examples": {},
        "primary": None,
        "baseline_comparisons": {},
        "model_id": config["model_id"],
        "seed": config["seed"],
        "initializations": {
            name: read_json(directory / name / "initialization-audit.json", {}) for name in ("main", "reward_only")
        },
        "novelty_weight": config["novelty_weight"],
        "finding": "No completed paired primary endpoint is available.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interval_scope": "Problem uncertainty conditional on one training seed; excludes seed variability.",
        "interval_caveat": (
            "Percentile bootstrap intervals can collapse at observed boundaries or when paired outcomes are identical. "
            "A zero-width interval does not prove zero population error or equivalence on unseen problems."
        ),
    }
    evaluations = {}
    for condition in ("original", "main", "reward_only"):
        training = experiment.get("conditions", {}).get(condition, {})
        evaluation = read_json(directory / condition / "evaluation.json", {
            "training_complete": condition == "original" or training.get("complete", False),
            "seconds": 0.0, "sets": {},
        })
        evaluations[condition] = evaluation
        method = {
            "training_complete": evaluation["training_complete"],
            "evaluation_seconds": evaluation["seconds"],
            "complete": False,
        }
        for split in ("test", "hard", "probes"):
            expected = config["probe_count"] if split == "probes" else manifest["splits"][split]["count"]
            method[split] = _split_summary(evaluation["sets"].get(split, []), expected, config["bootstrap_replicates"])
        groups = defaultdict(list)
        for row in evaluation["sets"].get("sampled", []):
            groups[row["problem_id"]].append(row)
        complete_groups = [rows for rows in groups.values() if len(rows) == config["samples_per_problem"]]
        for rows in complete_groups:
            if len({row["sample"] for row in rows}) != config["samples_per_problem"]:
                raise ValueError("Repeated sampled response indices")
        sampled = {
            "complete_problems": len(complete_groups),
            "expected": config["sampled_problem_count"],
            "generated_tokens": sum(row["generated_tokens"] for rows in groups.values() for row in rows),
            "partial_problems": len(groups) - len(complete_groups),
        }
        if complete_groups:
            counts = [sum(row["correct"] for row in rows) for rows in complete_groups]
            sampled.update(
                pass1=float(np.mean([pass_at_k(config["samples_per_problem"], c, 1) for c in counts])),
                pass8=float(np.mean([pass_at_k(config["samples_per_problem"], c, 8) for c in counts])),
                mean_unique_correct_structures=float(
                    np.mean([len({row["structure"] for row in rows if row["correct"]}) for rows in complete_groups])
                ),
            )
        method["sampled"] = sampled
        method["complete"] = bool(
            method["training_complete"]
            and all(method[s]["complete"] for s in ("test", "hard", "probes"))
            and sampled["complete_problems"] == sampled["expected"]
        )
        method["training"] = training
        generations = read_json(directory / condition / "metrics.json", [])
        method["training_observations"] = {
            "responses": sum(row["responses"] for row in generations),
            "generated_tokens": sum(row["generated_tokens"] for row in generations),
            "truncated": sum(row["truncated"] for row in generations),
            "weight_diagnostic": novelty_intervention(generations, config["novelty_weight"] if condition == "main" else 0.0),
        }
        phase = experiment.get("phases", {}).get(condition, {})
        method["training_seconds"] = phase.get("finished_at", time.time()) - phase["started_at"] if phase else 0.0
        summary["methods"][condition] = method
        # Fixed first problems expose matched successes and failures without selection by outcome.
        summary["examples"][condition] = evaluation["sets"].get("test", [])[:16]
    if all(
        name in summary["methods"]
        and summary["methods"][name]["training_complete"]
        and summary["methods"][name]["test"]["complete"]
        for name in ("main", "reward_only")
    ):
        primary = _paired_evaluations(evaluations["main"], evaluations["reward_only"], config["bootstrap_replicates"])
        summary["primary"] = primary
        if primary["difference"] <= 0:
            summary["finding"] = (
                "The directional improvement hypothesis was not supported in this seed: main accuracy did not exceed reward-only ES."
            )
        elif primary["low"] > 0:
            summary["finding"] = (
                "The paired test-problem interval supports an improvement conditional on this seed. Robustness across training seeds remains untested."
            )
        else:
            summary["finding"] = (
                "The point estimate favors the main method, but the paired interval includes zero; this pilot is inconclusive."
            )
        diagnostic = summary["methods"]["main"]["training_observations"]["weight_diagnostic"]
        if diagnostic["generations_observed"] and not diagnostic["generations_weight_changed"]:
            summary["finding"] += (
                " Adding novelty changed no main-condition update coefficients on the recorded populations; "
                "this pilot did not exercise an active novelty intervention."
            )
    # These comparisons were planned against the original model, but remain secondary.
    if summary["methods"]["original"]["test"]["complete"]:
        for condition in ("main", "reward_only"):
            method = summary["methods"][condition]
            if method["training_complete"] and method["test"]["complete"]:
                summary["baseline_comparisons"][condition] = _paired_evaluations(
                    evaluations[condition], evaluations["original"], config["bootstrap_replicates"]
                )
    summary["failures"] = experiment.get("failures", [])
    summary["protocol_amendments"] = experiment.get("protocol_amendments", [])
    summary["execution_repairs"] = experiment.get("execution_repairs", [])
    summary["recovery_audit"] = read_json(directory / "recovery-audit.json", {})
    summary["logged_generation_work"] = logged_generation_work(directory / "events.jsonl")
    summary["elapsed_seconds"] = (
        experiment.get("finished_at", time.time()) - experiment["started_at"] if "started_at" in experiment else 0
    )
    return summary


def _percent(value) -> str:
    """Render absent observations as missing, not numerical zero."""
    return "missing" if value is None else f"{100 * value:.2f}%"


def result_table(summary: dict) -> list[list[str]]:
    """One canonical table feeds Markdown, PDF and model cards."""
    table = [["Condition", "Test accuracy", "95% interval", "Hard accuracy", "pass@1", "pass@8", "Status"]]
    for name, method in summary["methods"].items():
        interval = method["test"].get("interval")
        table.append(
            [
                name,
                _percent(method["test"].get("accuracy")),
                " - ".join(map(_percent, interval)) if interval else "missing",
                _percent(method["hard"].get("accuracy")),
                _percent(method["sampled"].get("pass1")),
                _percent(method["sampled"].get("pass8")),
                "complete" if method["complete"] else "incomplete",
            ]
        )
    return table


def markdown_table(table: list[list[str]]) -> str:
    """Render generated numerical rows consistently across repository artifacts."""
    lines = ["| " + " | ".join(table[0]) + " |", "| " + " | ".join(["---"] * len(table[0])) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in table[1:])
    return "\n".join(lines)


def comparison_table(summary: dict) -> list[list[str]]:
    """State paired test differences in percentage points with explicit comparison roles."""
    table = [["Comparison", "Difference (pp)", "Paired 95% interval (pp)", "Problems"]]
    comparisons = {"Main - reward-only (primary)": summary["primary"]}
    comparisons.update({f"{name} - original (secondary)": result for name, result in summary["baseline_comparisons"].items()})
    for name, result in comparisons.items():
        if result:
            table.append([
                name, f"{100 * result['difference']:.3f}",
                f"[{100 * result['low']:.3f}, {100 * result['high']:.3f}]", str(result["n"]),
            ])
    return table


def diagnostic_table(summary: dict) -> list[list[str]]:
    """Keep probe performance and validity visible alongside the held-out endpoint."""
    table = [
        ["Condition", "Probe accuracy", "Test validity", "Hard validity", "Train tokens", "Eval tokens", "Train hours"]
    ]
    for name, method in summary["methods"].items():
        tokens = sum(method[s].get("generated_tokens", 0) for s in ("test", "hard", "probes", "sampled"))
        table.append(
            [
                name,
                _percent(method["probes"].get("accuracy")),
                _percent(method["test"].get("validity")),
                _percent(method["hard"].get("validity")),
                str(method["training_observations"]["generated_tokens"]),
                str(tokens),
                f"{method['training_seconds'] / 3600:.3f}",
            ]
        )
    return table


def model_description(summary: dict) -> str:
    """Read architecture from the recorded pinned-weight audit, never guessed model labels."""
    audit = next((item for item in summary["initializations"].values() if item), {})
    if not audit:
        return f"{summary['model_id']}; architecture audit not recorded."
    architecture = audit["architecture"]
    return (
        f"{summary['model_id']}: {architecture['num_hidden_layers']} layers, width {architecture['hidden_size']}, "
        f"{architecture['num_attention_heads']} query heads and {architecture['num_key_value_heads']} key/value heads; "
        f"{audit['parameter_elements']:,} unique parameter elements in {audit['unique_tensors']} tensors. "
        "Architecture and counts come from the pinned pretrained files and the generation-zero checkpoint audit."
    )


def diversity_table(summary: dict) -> list[list[str]]:
    """Expose within-problem sampled diversity separately from single greedy successes."""
    table = [["Condition", "Greedy correct structures", "Complete sampled problems", "Mean correct structures / sampled problem"]]
    for name, method in summary["methods"].items():
        value = method["sampled"].get("mean_unique_correct_structures")
        table.append([
            name, str(method["test"].get("correct_expression_structures", "missing")),
            str(method["sampled"]["complete_problems"]), "missing" if value is None else f"{value:.4f}",
        ])
    return table


def refresh_readme(path: Path, summary: dict, table: list[list[str]]) -> None:
    """Replace only the explicitly marked result block from the same recorded summary."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- BEGIN GENERATED PILOT RESULTS -->", "<!-- END GENERATED PILOT RESULTS -->"
    if start not in text or end not in text:
        return
    before, after = text.split(start, 1)[0], text.split(end, 1)[1]
    result = (
        f"\n{summary['finding']}\n\n{outcome_description(summary)}\n\n{markdown_table(table)}\n\n"
        f"{summary['interval_scope']} [Full report](reports/experiment-pilot.md).\n"
    )
    path.write_text(before + start + result + end + after, encoding="utf-8")


def outcome_description(summary: dict) -> str:
    """Explain observed benefits and failures without changing endpoints or selecting models."""
    methods = summary["methods"]
    if not all(methods[name]["training_complete"] and methods[name]["test"]["complete"] for name in methods):
        return "The paired final-checkpoint test comparison is still incomplete."
    original = methods["original"]
    trained = [methods[name] for name in ("main", "reward_only")]
    sentences = []
    if all(method["test"]["accuracy"] > original["test"]["accuracy"] for method in trained):
        sentences.append("Both trained checkpoints score higher than the original model on this test set; the main-versus-control comparison determines whether novelty adds a benefit.")
    if all(method["test"]["validity"] > original["test"]["validity"] for method in trained):
        sentences.append("Both also produce more legal expressions on the three/four-number test problems. Legality alone does not mean the target was reached.")
    if all(method["hard"]["complete"] for method in methods.values()):
        if all(method["hard"]["accuracy"] == 0 for method in trained):
            sentences.append("Neither trained checkpoint solves a five-number test problem in this evaluation.")
        if all(method["hard"]["validity"] < original["hard"]["validity"] for method in trained):
            sentences.append("Both have lower expression validity on those harder problems than the original model, so the validity improvement does not transfer uniformly.")
    return " ".join(sentences) or "The tables retain all observed outcomes; no broader performance claim follows from this single seed."


def intervention_description(summary: dict) -> str:
    """Describe whether the bonus changes update weights, without implying a trajectory replay."""
    sentences = [
        f"Descriptive diagnostic: compare the main arm's Z(Q) + {summary['novelty_weight']} N with Z(Q) on its recorded populations. "
        "The control computes novelty but applies coefficient zero, so its novelty-induced weight changes are zero. "
        "This local comparison does not simulate the counterfactual training trajectory."
    ]
    for name, method in summary["methods"].items():
        diagnostic = method["training_observations"]["weight_diagnostic"]
        if diagnostic["generations_observed"]:
            sentences.append(
                f"{name}: positive novelty in {diagnostic['candidates_positive_novelty']} of "
                f"{diagnostic['candidates_observed']} candidates; changed update weights in "
                f"{diagnostic['generations_weight_changed']} of {diagnostic['generations_observed']} generations "
                f"({diagnostic['candidates_weight_changed']} candidate weights)."
            )
    return " ".join(sentences)


def coverage_description(summary: dict) -> str:
    """Expose observed denominators so partial evaluation cannot look complete."""
    sentences = []
    for name, method in summary["methods"].items():
        splits = ", ".join(
            f"{split} {method[split]['n']}/{method[split]['expected']}" for split in ("test", "hard", "probes")
        )
        sampled = method["sampled"]
        sentences.append(
            f"{name}: greedy problems {splits}; sampled complete problems "
            f"{sampled['complete_problems']}/{sampled['expected']}."
        )
    return " ".join(sentences) or "No evaluation rows have been recorded."


def plot_results(directory: Path, summary: dict, output: Path) -> None:
    """Use zero-based axes with shared, data-driven limits across compared conditions.

    Tick units: https://matplotlib.org/stable/api/ticker_api.html .
    Each panel scales to its observed maximum; accuracy includes interval endpoints.
    """
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    colors = {"main": "#157f67", "reward_only": "#bc681e"}
    plotted = False
    keys = ("quality", "novelty", "accuracy")
    maxima = dict.fromkeys(keys, 0.0)
    last_generation = 1
    for name, color in colors.items():
        rows = read_json(directory / name / "metrics.json", [])
        for axis, key in zip(axes, keys, strict=True):
            if rows:
                axis.plot([r["generation"] for r in rows], [r[key] for r in rows], label=name, color=color, marker="o", markersize=3)
                maxima[key] = max(maxima[key], max(row[key] for row in rows))
                last_generation = max(last_generation, max(row["generation"] for row in rows))
                plotted = True
    for axis, key in zip(axes, keys, strict=True):
        bound = 2 if key == "novelty" else 1
        axis.set(xlabel="Generation", ylabel=key, ylim=(0, min(bound, max(0.001, 1.15 * maxima[key]))), xlim=(0.5, last_generation + 0.5))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        if key == "accuracy":
            axis.yaxis.set_major_formatter(PercentFormatter(1))
    if plotted:
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(
            0.5, 0.5, "No committed training observations", transform=axes[0].transAxes, ha="center", fontsize=8
        )
    fig.suptitle("Population means; perturbed candidates, not final test performance\nZero-based axes; scale differs by metric")
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 3.5), layout="constrained")
    names = [name for name, m in summary["methods"].items() if "accuracy" in m["test"]]
    upper = 1.0
    if names:
        means = np.array([summary["methods"][n]["test"]["accuracy"] for n in names])
        bounds = np.array([summary["methods"][n]["test"]["interval"] for n in names])
        upper = min(1, max(0.01, 1.15 * float(bounds.max())))
        axis.bar(
            names,
            means,
            color=[colors.get(n, "#607d9b") for n in names],
            yerr=np.maximum(0, np.vstack((means - bounds[:, 0], bounds[:, 1] - means))),
            capsize=5,
        )
    else:
        axis.text(0.5, 0.5, "No evaluation observations", transform=axis.transAxes, ha="center")
    axis.set(
        ylabel="Greedy exact accuracy", ylim=(0, upper), title="Held-out test accuracy; problem-bootstrap 95% intervals"
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)


def write_paper(path: Path, experiment: dict, summary: dict, figures: Path, table: list[list[str]]) -> None:
    """Build the dated paper from the same canonical results and registered configuration."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 7
    styles["Heading2"].keepWithNext = True
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))

    def section(title, body):
        paragraph(title, "Heading2")
        paragraph(body)

    config = experiment.get("config", read_json(Path("configs/pilot.json"), {}))
    paragraph("Correct-Response Novelty in Full-Weight LLM Evolution", "Title")
    paragraph('Libor "Burny" Burian', "Heading2")
    paragraph(datetime.now().date().isoformat())
    section(
        "Abstract",
        "We test an adaptation of historical response novelty to full-weight Gaussian evolution strategies on witnessed arithmetic problems. "
        "The comparison uses one paired training seed and an untrained base model. "
        + html.escape(summary["finding"])
        + " Problem-resampling intervals exclude training-seed variability. All tables are generated from persisted evaluations; incomplete outcomes are labeled.",
    )
    section(
        "Context, motivation and introduction",
        "The experiment asks whether exploration among correct answer forms improves held-out task success. "
        "ExTra [1] supplies the historical semantic novelty concept; ES at Scale [2] supplies the full-weight optimization reference. "
        "Their combination uses population-level ES rather than GRPO, with a different sibling population and task, so it is an adaptation rather than a reproduction. "
        "A reward-only ES control is necessary because ES can itself change solution coverage [3].",
    )
    section(
        "Background and related work",
        "Novelty-seeking ES was previously studied in control [4]. LLM-specific alternatives include EVOL-RL's within-group novelty [5], "
        "DARLING's quality/diversity objective [6], and the model/task coevolution of AC/DC [7]. These use different training or population mechanisms. "
        "The repository literature audit distinguishes prompt evolution, artifact search and actual model-parameter training, with code/license verification.",
    )
    section(
        "Method",
        f"The model is {html.escape(config['model_id'])}, revision {config['model_revision']}. "
        f"Each generation evaluates {config['population']} independent Gaussian candidates (sigma {config['sigma']}) "
        f"on {config['rotating_count']} rotating training problems and {config['probe_count']} fixed training probes. "
        f"Population-z-scored task reward is combined with {config['novelty_weight']} times correctness-gated cosine novelty for the main condition. "
        f"The archive retains {config['memory_size']} correct embeddings per probe after seeded insertion. "
        f"These coefficients weight noise directly, without another normalization; effective alpha is {config['alpha']}. "
        "FP32 masters reconstruct BF16 candidates afresh and tied tensors are updated once. Prefix regeneration and backpropagation are absent.",
    )
    paragraph(html.escape(model_description(summary)))
    paragraph(
        "For candidate i, Q_i is the mean response reward: exact success receives 1, a legal expression using every supplied "
        "number but missing the target receives 0.1, and an invalid expression receives 0. "
        "For probe p, n_i,p = I(correct_i,p) [1 - max cosine(e_i,p, z)]. The reference set consists of correct historical "
        "responses and correct responses from other candidates on that same probe; self-comparisons are excluded. "
        "An empty set has maximum zero. N_i is the probe mean. History is frozen while scoring the whole generation, "
        "then correct embeddings are appended in seeded shuffled order and the oldest entries are removed."
    )
    paragraph(
        "Z_i = (Q_i - mean(Q)) / (std_population(Q) + 1e-8), with zero Z for constant Q. "
        f"W_i = Z_i + {config['novelty_weight']} N_i for the main arm and W_i = Z_i for control. "
        "The update is theta_next = theta + (alpha / population) sum_i W_i epsilon_i. "
        "Each epsilon_i is independent standard Gaussian noise over every unique parameter tensor. "
        "A common positive novelty bonus is retained. Alpha includes the inverse-noise factor; it is not "
        "an additional alpha/sigma update. MiniLM is frozen, and both conditions compute the same categories of probe work."
    )
    paragraph(
        "Protocol amendment A replaced combined-reward ranks with source-aligned z-scored quality plus post-normalization novelty. "
        "It was published after calibration and partial original-model evaluation were visible, but before either trained condition began. "
        "Both arms start from the original pretrained weights using the amended rule from their first generation. "
        "The original registration, changed configuration, code revisions and unchanged absolute deadline are retained in the run record. "
        "This ordering follows ES at Scale Algorithm 2 and ExTra Equation 2; it is not evidence that the hybrid outperforms the abandoned rule."
    )
    section(
        "Data, evaluation and preregistered analysis",
        "Synthetic problems have exact rational witnesses. Global number-multiset isolation prevents permutations or alternative targets crossing splits. "
        "Only constants supplied in the prompt and allowlisted arithmetic are accepted. Greedy test accuracy at the final scheduled checkpoint is primary. "
        "The directional hypothesis is falsified for this pilot if main minus control accuracy is nonpositive. "
        "A paired problem bootstrap quantifies conditional uncertainty. Sampled pass@k follows the unbiased subset estimator [8]. "
        "Validation cannot select checkpoints. Exact dataset hashes, schedules, timing rules and sampling settings are recorded in reports/preregistration.md and configs/pilot.json.",
    )
    manifest = experiment.get("dataset_manifest", read_json(Path("data/manifest.json"), {}))
    counts = manifest.get("splits", {})
    paragraph(
        "Dataset counts: " + ", ".join(f"{name} {item['count']}" for name, item in counts.items()) + ". "
        f"Greedy responses allow at most {config['max_new_tokens']} new tokens. "
        f"The sampled evaluation uses {config['samples_per_problem']} responses on each of "
        f"{config['sampled_problem_count']} fixed test problems, with temperature {config['sampled_temperature']}, "
        f"top-p {config['sampled_top_p']} and top-k disabled. pass@k = 1 - C(n-c,k)/C(n,k), where c is the number "
        f"of correct samples. Intervals use {config['bootstrap_replicates']} paired resamples, seed {config['data_seed']}."
    )
    section("Results and claims", html.escape(summary["finding"]))
    cells = [[Paragraph(html.escape(cell), styles["BodyText"]) for cell in row] for row in table]
    result = Table(cells, colWidths=[64, 67, 87, 64, 52, 52, 66], repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]
        )
    )
    story.extend([result, Spacer(1, 12)])
    paragraph(html.escape(coverage_description(summary)))
    primary = summary.get("primary")
    if primary:
        paragraph(
            f"Main minus control: {primary['difference'] * 100:.3f} percentage points; paired 95% interval "
            f"[{primary['low'] * 100:.3f}, {primary['high'] * 100:.3f}], on {primary['n']} paired problems."
        )
    comparisons = comparison_table(summary)
    if len(comparisons) > 1:
        paired = Table(
            [[Paragraph(html.escape(cell), styles["BodyText"]) for cell in row] for row in comparisons],
            colWidths=[174, 82, 140, 56], repeatRows=1,
        )
        paired.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ]))
        story.extend([paired, Spacer(1, 8)])
        paragraph("Comparisons with the original checkpoint are descriptive secondary estimates with unadjusted intervals.")
    story.append(Image(str(figures / "accuracy.png"), width=6.0 * inch, height=3.0 * inch))
    paragraph(html.escape(summary["interval_caveat"]))
    paragraph("Probe performance, validity and resources", "Heading2")
    diagnostics = Table(
        [[Paragraph(html.escape(cell), styles["BodyText"]) for cell in row] for row in diagnostic_table(summary)],
        colWidths=[64, 64, 64, 64, 68, 68, 60],
        repeatRows=1,
    )
    diagnostics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]
        )
    )
    story.extend([diagnostics, Spacer(1, 8)])
    paragraph("Within-problem expression diversity", "Heading2")
    diversity = Table(
        [[Paragraph(html.escape(cell), styles["BodyText"]) for cell in row] for row in diversity_table(summary)],
        colWidths=[80, 110, 110, 152], repeatRows=1,
    )
    diversity.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.extend([diversity, Spacer(1, 8)])
    paragraph(
        "Greedy structure counts equal the number of successful test problems because only one answer is generated per problem. "
        "The sampled statistic counts distinct correct canonical ASTs within each problem before averaging, including zeros. "
        "It measures syntactic diversity, not algebraic or semantic equivalence."
    )
    paragraph(
        "Train tokens count candidates in committed generations; evaluation tokens count persisted evaluation responses. "
        "Calibration, validation, interrupted batches and replay costs are excluded from these two table columns. "
        "Training hours are wall-clock phase duration and include checkpointing. Probes are training examples, not held-out data."
    )
    section("Did the novelty bonus affect optimization?", html.escape(intervention_description(summary)))
    section(
        "Training dynamics and output inspection",
        "Population metrics describe perturbed candidates, not the master checkpoint. "
        "Reward is the optimization signal; no cross-entropy loss is optimized in this pilot. "
        "The interactive result page shows the same first test problems across methods, including failures, without choosing examples by outcome.",
    )
    story.append(Image(str(figures / "training.png"), width=6.2 * inch, height=1.92 * inch))
    for name, examples in summary["examples"].items():
        if examples:
            row = examples[0]
            section(
                "Fixed example: " + name, html.escape(row["prompt"]) + "<br/>Output: " + html.escape(row["response"])
            )
    section(
        "Implications and discussion",
        html.escape(outcome_description(summary)) + " "
        "This pilot is a controlled objective intervention under one initialization/seed schedule. "
        "It can identify a conditional difference or a feasibility failure, but cannot establish a general advantage of novelty training. "
        "Any improvement must be checked across additional seeds and separated from probe memorization. "
        "A rise in novelty alone, or a selected qualitative example, is insufficient evidence.",
    )
    section(
        "Limitations",
        "Training-seed sensitivity is unmeasured. Embedding similarity is a proxy and may reward irrelevant wording; "
        "canonical expression trees are an independent syntactic diagnostic, not full algebraic equivalence. "
        "Synthetic arithmetic does not measure general language quality. Similar pretraining patterns cannot be excluded. "
        "The fixed-probe objective can overfit. Small time budgets, laptop thermal variability and batch-level floating-point behavior limit extrapolation. "
        "CPU tests, report generation, audits and backups ran alongside training; elapsed times are descriptive, not isolated systems benchmarks. "
        "Secondary endpoints are descriptive; no multiple-comparison significance claims are made.",
    )
    section(
        "Conclusion",
        html.escape(summary["finding"])
        + " Multi-seed studies and archive ablations require a separately frozen amendment after this result is published.",
    )
    section(
        "Appendix: reproducibility and failures",
        "Run uv sync --locked, uv run pytest, uv run novelty-es prepare, "
        "uv run novelty-es download, and uv run novelty-es run. Repeating run resumes the same clock. "
        "uv run novelty-es report regenerates tables, figures, this paper and the static website. "
        "Atomic checkpoints contain weights, all RNG streams, archive, cursor, config and deadline. "
        f"Recorded technical failures: {len(summary['failures'])}. Experiment elapsed seconds: {summary['elapsed_seconds']:.3f}. "
        "Exact exceptions and incomplete conditions are retained in the generated Markdown report and JSON artifacts.",
    )
    environment = experiment.get("environment", {})
    calibration = experiment.get("calibration", {})
    paragraph(
        "Execution provenance: "
        + html.escape(str(environment.get("gpu", "not executed")))
        + "; Python "
        + html.escape(str(environment.get("python", "not recorded")))
        + ". Initial execution revision: "
        + html.escape(experiment.get("code_revision", "not recorded"))
        + ". Training implementation revision: "
        + html.escape(experiment.get("training_code_revision", experiment.get("code_revision", "not recorded")))
        + ". "
        + f"Frozen generations per arm: {calibration.get('generations', 'not calibrated')}; "
        + f"inference batch: {calibration.get('batch_size', 'not calibrated')}. "
        + "The timing rule uses the slowest calibration candidate, a complete update and checkpoint, and amortized "
        + "validation cost. The budget reserves a fraction of each training phase and freezes the common count before training."
    )
    paragraph(
        "Embedding revision: "
        + config["embedding_revision"]
        + ". ExTra upstream revision: "
        + config["extra_revision"]
        + ". Complete dependency versions, dataset hashes, absolute deadline, per-phase timings, token-work accounting, "
        + "raw outputs and completion flags are machine-readable in the published experiment artifacts. "
        + "Full-weight noise is regenerated by a hash of perturbation seed and tensor name using an explicit PyTorch generator. "
        + "Bit-level replay is scoped to the recorded hardware and software environment."
    )
    recovery = summary["recovery_audit"]
    if recovery:
        paragraph(
            f"The observed storage-failure recovery regenerated a batch of {recovery['replayed_batch_rows']} sampled responses. "
            "Decoded text, token IDs, identities and verifier fields matched the preserved failed-write snapshot exactly; "
            "only elapsed batch time was excluded. The source hash and machine-readable check are published with the artifacts. "
            "This is evidence for this evaluation replay, not arbitrary training restarts or environments."
        )
    references = [
        ("1. ExTra", "https://arxiv.org/abs/2606.24994"),
        ("2. Evolution Strategies at Scale", "https://arxiv.org/abs/2509.24372"),
        ("3. Beyond the Best Guess", "https://arxiv.org/abs/2608.12679"),
        ("4. Novelty-seeking ES", "https://arxiv.org/abs/1712.06560"),
        ("5. EVOL-RL", "https://arxiv.org/abs/2509.15194"),
        ("6. DARLING", "https://arxiv.org/abs/2509.02534"),
        ("7. AC/DC", "https://arxiv.org/abs/2604.14969"),
        ("8. Evaluating Large Language Models Trained on Code", "https://arxiv.org/abs/2107.03374"),
        (
            "Research protocol and source",
            "https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training",
        ),
    ]
    paragraph("References", "Heading2")
    for label, url in references:
        paragraph(f'<link href="{url}" color="#185b96">{html.escape(label)}</link>')
    section(
        "AI statement",
        "Codex wrote this codebase and paper, and human cleanup and rewriting are in progress. "
        "A human provided the exact specification that was executed.",
    )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(45, 25, 'Libor "Burny" Burian | Novelty ES pilot')
        canvas.drawRightString(565, 25, str(doc.page))

    SimpleDocTemplate(
        str(path),
        rightMargin=42,
        leftMargin=42,
        topMargin=38,
        bottomMargin=40,
        title="Correct-Response Novelty in Full-Weight LLM Evolution",
        author='Libor "Burny" Burian',
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def report(directory: Path, output_root: Path = Path(".")) -> dict:
    """Single command regenerates JSON, tables, plots, reports, model cards, Pages and PDF."""
    summary = summarize(directory)
    atomic_json(directory / "summary.json", summary)
    experiment = read_json(directory / "experiment.json", {})
    table = result_table(summary)
    reports, figures = output_root / "reports", output_root / "reports/figures"
    reports.mkdir(parents=True, exist_ok=True)
    plot_results(directory, summary, figures)
    primary = summary["primary"]
    comparison = (
        f"Paired difference: {primary['difference'] * 100:.3f} percentage points; 95% interval "
        f"[{primary['low'] * 100:.3f}, {primary['high'] * 100:.3f}]; {primary['n']} paired test problems."
        if primary
        else "The registered paired endpoint is incomplete or not yet available."
    )
    text = (
        f'# Pilot experiment\n\nLibor "Burny" Burian · {datetime.now().date().isoformat()}\n\n'
        f"{summary['finding']}\n\n{comparison}\n\n{summary['interval_scope']}\n\n"
        f"{outcome_description(summary)}\n\n"
        f"{summary['interval_caveat']}\n\n"
        f"{markdown_table(table)}\n\n{coverage_description(summary)}\n\n"
        f"{markdown_table(comparison_table(summary))}\n\n"
        "Comparisons with the original checkpoint are descriptive secondary estimates with unadjusted intervals.\n\n"
        "![Accuracy](figures/accuracy.png)\n\n![Training](figures/training.png)\n\n"
        f"## Probe performance, validity and resources\n\n{markdown_table(diagnostic_table(summary))}\n\n"
        "Train tokens count only candidates in committed generations; evaluation tokens count persisted evaluation responses. "
        "Calibration, validation, interrupted batches and replay costs are excluded from those columns. "
        "Training hours include phase overhead and checkpointing. Probes are training examples, not held-out data.\n\n"
        f"## Novelty intervention diagnostic\n\n{intervention_description(summary)}\n\n"
        "See [analysis notes](analysis-notes.md) for when this descriptive check was added.\n\n"
        f"## Within-problem expression diversity\n\n{markdown_table(diversity_table(summary))}\n\n"
        "Greedy structure counts equal successful problems, since each problem has one response. "
        "Sampled diversity counts distinct correct canonical ASTs within each problem, then averages over complete sampled groups, including zeros. "
        "This is syntactic diversity; algebraically equivalent expressions may count separately.\n\n"
        "## Method and scope\n\nSee the [preregistration](preregistration.md), [configuration](../configs/pilot.json), "
        "[data hashes](../data/manifest.json), and [literature audit](../docs/literature.md). "
        "This is an adaptation of [ExTra](https://arxiv.org/abs/2606.24994) and "
        "[ES at Scale](https://arxiv.org/abs/2509.24372), not a reproduction. "
        "The archive gate uses exact correctness; positive legality reward is insufficient. "
        "Validation does not choose the final checkpoint. "
        "Amendment A replaces combined-reward ranks with population-z-scored task reward plus post-normalization novelty. "
        "It was published after calibration and partial original-model evaluation, before either trained condition. "
        "Both trained arms start from original pretrained weights using the amended rule from generation one.\n\n"
        + model_description(summary) + "\n\n"
        "## Recorded measurements\n\nAll values below come from persisted outcomes. Missing conditions remain missing. "
        "Population training accuracy is distinct from unperturbed master validation and held-out test accuracy.\n\n"
        "```json\n"
        + json.dumps(
            {
                "methods": summary["methods"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "calibration": experiment.get("calibration", {}),
                "logged_generation_work": summary["logged_generation_work"],
            },
            indent=2,
        )
        + "\n```\n\n"
        "## Failures, deviations and learnings\n\n```json\n" + json.dumps(
            {
                "failures": summary["failures"],
                "protocol_amendments": summary["protocol_amendments"],
                "execution_repairs": summary["execution_repairs"],
                "recovery_audit": summary["recovery_audit"],
                "initial_code_revision": experiment.get("code_revision"),
                "training_code_revision": experiment.get("training_code_revision", experiment.get("code_revision")),
            }, indent=2,
        ) + "\n```\n\n"
        "The first audit reconciled ExTra's correctness-filtered paper specification with its unfiltered embedding-memory insertion. "
        "Explicit Boolean masking and separate score/append operations implement the intended experiment. "
        "FP32 master reconstruction avoids repeated BF16 perturbation subtraction. "
        "Tests exercise interrupted candidate replay, archive RNG restoration, deadline preservation and exact arithmetic. "
        "Hyperparameters are preregistered choices, not claimed optima. Further training requires reviewing this pilot.\n\n"
        "## Limitations\n\nOne seed; synthetic arithmetic; embedding-proxy and probe-overfitting risks; "
        "no general-language benchmark; descriptive secondary analyses. Bootstrap intervals omit seed variability. "
        "CPU tests, reporting, audits and backups overlapped training, so timing is not an isolated systems benchmark.\n"
    )
    (reports / "experiment-pilot.md").write_text(text, encoding="utf-8")
    refresh_readme(output_root / "README.md", summary, table)
    (reports / "all-experiments.md").write_text(
        "# All experiments\n\nThe current registered study is the paired pretrained pilot. "
        "No scratch or diagnostic run is represented as executed.\n\n"
        + summary["finding"] + "\n\n" + outcome_description(summary) + "\n\n"
        + markdown_table(table)
        + "\n\n" + model_description(summary)
        + "\n\n" + markdown_table([
            ["Condition", "Parameter update", "Objective", "Training data / schedule"],
            ["Original", "Frozen pretrained checkpoint", "No optimization here", "No training here"],
            ["Main", "Full-weight Gaussian ES", f"Z(task reward) + {summary['novelty_weight']} novelty", "Registered Countdown split; rotating problems and fixed probes"],
            ["Reward-only", "Full-weight Gaussian ES", "Z(task reward)", "Same problems, perturbation seeds and generation count"],
        ])
        + "\n\n[Full pilot report](experiment-pilot.md). [Deferred studies](../docs/roadmap.md).\n\n"
        "ES means comparing random weight changes instead of using backpropagation. Novelty is distance from previous "
        "correct answers. Quality is verified arithmetic reward. Accuracy is exact success frequency. "
        "pass@k estimates the chance of at least one success among k samples. A confidence interval here describes "
        "uncertainty from which test problems were sampled, not uncertainty over training seeds.\n",
        encoding="utf-8",
    )
    site = output_root / "docs/site"
    site.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("web/index.html", site / "index.html")
    atomic_json(site / "status.json", dashboard_state(directory))
    (site / ".nojekyll").touch()
    for condition in ("main", "reward_only"):
        model_dir = directory / condition / "model"
        if model_dir.exists():
            config = experiment["config"]
            (model_dir / "README.md").write_text(
                f"---\nlicense: apache-2.0\nbase_model: {config['model_id']}\nlibrary_name: transformers\n"
                "pipeline_tag: text-generation\ntags:\n- evolution-strategies\n- countdown\n---\n\n"
                f"# Novelty ES pilot: {condition}\n\n"
                + summary["finding"]
                + "\n\n" + outcome_description(summary) + "\n\n"
                + markdown_table(table)
                + "\n\nResearch-only arithmetic adaptation, with no evidence of general language improvements. "
                "Intervals exclude training-seed variability. See the full preregistration, source, provenance and limitations at "
                "https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training .\n\n"
                f"Base revision: `{config['model_revision']}`. Training and evaluation use the recorded seed and configuration.\n",
                encoding="utf-8",
            )
    # The artifact repository is a documented research release, not a flat example dataset.
    # Hub card format: https://huggingface.co/docs/hub/datasets-cards .
    (directory / "README.md").write_text(
        "---\nlicense: apache-2.0\nlanguage:\n- en\ntags:\n- evolution-strategies\n- reproducibility\n---\n\n"
        "# Novelty ES pilot evidence\n\n" + summary["finding"] + "\n\n" + markdown_table(table)
        + "\n\n" + summary["interval_scope"] + "\n\n"
        "This repository contains raw evaluations, timestamped generation traces, configuration and code provenance, "
        "initialization and recovery audits, and the final resumable FP32 training checkpoints. "
        "Inference-ready BF16 models are separate items in the same Hugging Face collection. "
        "The checkpoints contain master tensors, RNG states, archive, cursor and the original absolute deadline. "
        "They preserve the completed run; loading them does not grant a new training allowance. "
        "Use the project's weights-only checkpoint loader in the recorded environment. "
        "The SHA-256 manifest covers the explicitly uploaded evidence files. "
        "The final upload receipt records immutable Hub commit revisions.\n\n"
        "The source, synthetic datasets and their hashes, frozen configuration, dependency lock, detailed report and PDF are at "
        "https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training . "
        "Run `uv run novelty-es report --run-dir PATH` from that checkout to regenerate the report from these artifacts. "
        "The method adapts [ExTra](https://arxiv.org/abs/2606.24994) and "
        "[ES at Scale](https://arxiv.org/abs/2509.24372); it is not a reproduction. "
        "Amendment A preceded all training updates but followed calibration and partial original-model evaluation. "
        "This single-seed arithmetic study provides no evidence about general language quality or seed robustness.\n",
        encoding="utf-8",
    )
    shutil.copyfile("LICENSE", directory / "LICENSE")
    write_paper(output_root / "output/pdf/novelty-es-pilot.pdf", experiment, summary, figures, table)
    print(json.dumps({"report": str(reports / "experiment-pilot.md"), "finding": summary["finding"]}, indent=2))
    return summary
