"""Verify both staged papers, dashboards, cards, figures and tables against fresh calculations."""

import argparse
import json
import subprocess
from pathlib import Path

from pypdf import PdfReader
from verify_presentations import check_study, equal

from novelty_es.audit_reporting import EVIDENCE, REVIEW, audit_description
from novelty_es.verification import read, sha256


def verify_study(key, raw, stage, verified, evidence):
    """Validate all historical summary tables and independent strict estimators."""
    data = evidence["studies"][key]
    outcome = check_study(raw, stage, verified["studies"][key])
    slug = "pilot" if key == "pilot" else "continuation-seed0-gen100"
    release = stage / "releases" / slug
    summary = read(release / "summary.json")
    audit = summary["scientific_audit"]
    if not equal(audit["strict_primary"], data["comparisons"]["primary"]):
        raise ValueError(f"{key} strict comparison differs between evidence and report")
    for arm, method in summary["methods"].items():
        for split, strict in data["evaluations"][arm].items():
            if split == "sampled":
                for k in ("pass1", "pass8", "mean_unique_correct_structures"):
                    if not 0 <= strict[k] <= (16 if k == "mean_unique_correct_structures" else 1):
                        raise ValueError(f"{arm} {k} outside support")
            elif not equal(method[split]["accuracy"], strict["historical_correct"] / strict["responses"]):
                raise ValueError(f"{arm}/{split} historical accuracy mismatch")
    note = audit_description(audit)
    report = stage / "reports" / f"experiment-{slug}.md"
    cards = [release / "README.md", release / "main/model/README.md", release / "reward_only/model/README.md"]
    for path in [report, stage / "README.md", stage / "reports/all-experiments.md", *cards]:
        if note not in path.read_text(encoding="utf-8"):
            raise ValueError(f"Grading correction missing from {path}")
    paper = stage / "output/pdf" / f"novelty-es-{slug}.pdf"
    pages = PdfReader(paper).pages
    text = " ".join(" ".join(page.extract_text().split()) for page in pages)
    for phrase in (
        "Independent grading and paired sensitivity analysis",
        "repetition penalty",
        "McNemar",
        "AI statement",
    ):
        if phrase not in text:
            raise ValueError(f"{phrase} missing from paper")
    figures = (stage / "reports/figures") / (slug if key != "pilot" else "")
    outcome.update(
        paper_pages=len(pages),
        pdf_sha256=sha256(paper),
        report_sha256=sha256(report),
        figures={p.name: sha256(p) for p in figures.glob("*.png")},
        card_hashes={p.relative_to(release).as_posix(): sha256(p) for p in cards},
    )
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("pilot-dir", "continuation-dir", "verification", "release-dir", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    a = parser.parse_args()
    full = read(a.verification)
    stage = a.release_dir
    evidence = read(stage / EVIDENCE)
    result = {"studies": {}}
    for key, raw in (("pilot", a.pilot_dir), ("continuation", a.continuation_dir)):
        result["studies"][key] = verify_study(key, raw, stage, full, evidence)
    result["review_sha256"] = sha256(stage / REVIEW)
    result["evidence_sha256"] = sha256(stage / EVIDENCE)
    # This file index distinguishes reviewed presentations from source evidence and immutable historical claims.
    paths = (
        subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        .decode()
        .split("\0")
    )
    result["surface_index"] = {}
    for name in sorted(filter(None, paths)):
        if name == a.output.as_posix() or name.startswith("reports/reviews/publication-"):
            continue
        if (
            name.startswith(("docs/site/", "reports/figures/", "output/pdf/", "reports/summaries/", "web/"))
            or name == "README.md"
        ):
            role = "Published presentation of the matched endpoints, split metrics and limitations"
        elif (
            name.startswith("reports/")
            and (
                "audit" in name.lower() or "independent-" in name or "claim-ledger" in name or "claim-inventory" in name
            )
            and not name.startswith("reports/reviews/")
        ):
            role = "Historical assertions; use the current claim map and original records for evidence"
        elif name.startswith(("data/", "configs/", "reports/preregistration")):
            role = "Frozen protocol, settings or input evidence; checked against public commits and content hashes"
        elif name.startswith(("tests/", "scripts/", "src/")):
            role = "Implementation, diagnostics or test fixture; capability claims depend on the raw-record checks"
        elif name.startswith(("docs/", "reports/")) or name in ("NOTICE", "AGENTS.md"):
            role = "Project documentation, research record or attribution; see the claim map and primary references"
        else:
            role = "Build configuration, dependency or license metadata"
        result["surface_index"][name] = role
    a.output.parent.mkdir(exist_ok=True, parents=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Both staged studies agree with the historical checks and the independent registered-rule reconstruction.")


if __name__ == "__main__":
    main()
