"""Verify server timestamps, Git ancestry and each file of the six original Hub revisions."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

from novelty_es.verification import read

REPOSITORY = "BurnyCoder/evolutionary-strategies-novelty-search-llm-training"


def git_blob(path):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def github_events(pilot, recovery, audit):
    """A public merge time and ancestry jointly bound public availability of the document."""
    p, c = (read(d / "experiment.json") for d in (pilot, recovery))
    interrupted = read(recovery / "restart-parent/experiment.json")
    for rev, path, value in (
        ("81fcbb9", "data/manifest.json", p["dataset_manifest"]),
        ("f25f171", "data/continuation/manifest.json", c["config"]["continuation"]["fresh_manifest"]),
        ("81fcbb9", "configs/pilot.json", p["initial_registered_config"]),
        ("158becd", "configs/pilot.json", p["config"]),
        ("f25f171", "configs/continuation.json", interrupted["config"]),
        ("f874d78", "configs/continuation-recovery.json", c["config"]),
    ):
        frozen = json.loads(subprocess.check_output(["git", "show", f"{rev}:{path}"]))
        if frozen != value:
            raise ValueError(f"Run metadata differs from registration {rev}:{path}")
    checks = [
        (1, "81fcbb9", p["started_at"], "pilot registration and core"),
        (7, "158becd", p["phases"]["main"]["started_at"], "weighting amendment"),
        (19, "f25f171", interrupted["started_at"], "continuation registration"),
        (20, None, interrupted["started_at"], "continuation core"),
        (21, "f874d78", c["started_at"], "recovery registration"),
    ]
    result = []
    for pr, registration, start, label in checks:
        item = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}/pulls/{pr}"]))
        merged = datetime.fromisoformat(item["merged_at"]).timestamp()
        if merged >= start:
            raise ValueError(f"{label} public merge did not precede execution")
        if registration:
            subprocess.run(["git", "merge-base", "--is-ancestor", registration, item["merge_commit_sha"]], check=True)
        result.append(
            dict(
                event=label,
                url=item["html_url"],
                merged_at=item["merged_at"],
                merge_commit=item["merge_commit_sha"],
                registration=registration,
                execution_start=start,
                margin_seconds=start - merged,
            )
        )
    first_baseline = audit["studies"]["pilot"]["trace"]["phase_prompt_times"]["evaluation/original/test"]["first"]
    if datetime.fromisoformat(first_baseline) >= datetime.fromisoformat(result[1]["merged_at"]):
        raise ValueError("Amendment timing relative to baseline differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("pilot-dir", "continuation-dir", "verification", "audit", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    full = read(args.verification)
    if not full["source_files_unchanged"] or not full["reused_original_evaluations_match"]:
        raise ValueError("Historical raw-record comparison incomplete")
    recovery = read(args.continuation_dir / "experiment.json")["recovery"]
    for name, pinned in recovery["files"].items():
        if full["input_files"]["continuation"][f"restart-parent/{name}"] != pinned:
            raise ValueError(f"Restart-parent evidence differs: {name}")
    result = dict(github=github_events(args.pilot_dir, args.continuation_dir, read(args.audit)), hub=[])
    api = HfApi()
    for key, root, receipt in (
        ("pilot", args.pilot_dir, "publication.json"),
        ("continuation", args.continuation_dir, "publication-continuation-seed0-gen100.json"),
    ):
        pub = read(Path("reports") / receipt)
        times = read(root / "experiment.json")
        for role, url in {**pub["models"], "artifacts": pub["artifacts"]}.items():
            dataset = role == "artifacts"
            kind = "dataset" if dataset else "model"
            repo = url.removeprefix("https://huggingface.co/").removeprefix("datasets/")
            revision = pub["revisions"][role]
            info = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError("Hub revision mismatch")
            files = {}
            for entry in info.siblings:
                rel = Path(entry.rfilename) if dataset else Path(role) / "model" / entry.rfilename
                signature = full["input_files"][key][rel.as_posix()]
                if signature["bytes"] != entry.size or (
                    entry.lfs
                    and entry.lfs.sha256 != signature["sha256"]
                    or not entry.lfs
                    and git_blob(root / rel) != entry.blob_id
                ):
                    raise ValueError(f"Local bytes differ from {repo}@{revision}/{entry.rfilename}")
                files[entry.rfilename] = signature
            created = api.list_repo_commits(repo, repo_type=kind, revision=revision)[0].created_at
            result["hub"].append(
                dict(
                    url=url + "/tree/" + revision,
                    revision=revision,
                    files=files,
                    created_at=created.isoformat(),
                    seconds_after_final_phase=created.timestamp() - times["phases"]["final"]["deadline"],
                    seconds_before_overall_deadline=times["deadline"] - created.timestamp(),
                )
            )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
