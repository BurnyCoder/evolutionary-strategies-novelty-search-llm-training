"""Verify immutable Hub uploads against local files, without running either model.

Hub metadata supplies SHA-256 for large files; small files are downloaded at the
recorded commit and hashed directly. API reference:
https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api .
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv


def digest(path):
    """Stream file hashes so full-weight checkpoints do not require extra RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    """Require all three releases, exact file coverage, matching hashes and collection membership."""
    load_dotenv()
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/pilot"))
    args = parser.parse_args()
    directory = args.run_dir
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    continued = "continuation" in experiment
    slug = experiment["config"]["experiment"] if continued else "pilot"
    published = json.loads((directory / "published.json").read_text(encoding="utf-8"))
    assert set(published["models"]) == {"main", "reward_only"} and published["artifacts"], "Publication is unfinished"
    manifest = json.loads((directory / "artifact-manifest.json").read_text(encoding="utf-8"))
    api = HfApi()
    verified = {}
    expected_items = set()
    for name in ("main", "reward_only", "artifacts"):
        kind = "dataset" if name == "artifacts" else "model"
        url = published["artifacts"] if kind == "dataset" else published["models"][name]
        repo_id = "/".join(url.rstrip("/").split("/")[-2:])
        revision = published["revisions"][name]
        folder = directory if kind == "dataset" else directory / name / "model"
        files = set(manifest["files"]) | {"artifact-manifest.json"} if kind == "dataset" else {p.name for p in folder.iterdir() if p.is_file()}
        info = api.repo_info(repo_id, repo_type=kind, revision=revision, files_metadata=True)
        assert info.sha == revision
        remote = {entry.rfilename: entry for entry in info.siblings}
        assert set(remote) - {".gitattributes"} == files, f"Unexpected remote file set: {name}"
        checked = {}
        for filename in sorted(files):
            path, metadata = folder / filename, remote[filename]
            expected = digest(path)
            assert path.stat().st_size == metadata.size
            if kind == "dataset" and filename != "artifact-manifest.json":
                assert expected == manifest["files"][filename]["sha256"]
            if metadata.lfs:
                assert expected == metadata.lfs.sha256
            else:
                downloaded = Path(hf_hub_download(repo_id, filename, repo_type=kind, revision=revision))
                assert digest(downloaded) == expected
            checked[filename] = {"sha256": expected, "bytes": metadata.size}
        verified[name] = {"url": url, "revision": revision, "files": checked}
        expected_items.add((repo_id, kind))
    collection = api.get_collection(published["collection"].split("/collections/", 1)[1])
    assert expected_items <= {(item.item_id, item.item_type) for item in collection.items}
    pages = "https://burnycoder.github.io/evolutionary-strategies-novelty-search-llm-training/"
    local_site = Path("docs/site")
    if continued:
        pages += slug + "/"
        local_site /= slug
    with urlopen(pages + "status.json", timeout=30) as response:
        remote_status = json.load(response)
    assert remote_status == json.loads((local_site / "status.json").read_text(encoding="utf-8")), "Published Pages results differ"
    now = time.time()
    result = {
        "verified_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "elapsed_since_start_seconds": now - experiment["started_at"],
        "clock_origin": (
            "recovery initialization" if "recovery" in experiment
            else "continuation initialization" if continued else "GPU calibration"
        ),
        "absolute_deadline": experiment["deadline"],
        "final_phase_deadline": experiment["phases"]["final"]["deadline"],
        "within_absolute_allowance": now <= experiment["deadline"],
        "within_final_phase_allowance": now <= experiment["phases"]["final"]["deadline"],
        "final_phase_overrun_seconds": max(0.0, now - experiment["phases"]["final"]["deadline"]),
        "collection": published["collection"], "repositories": verified,
        "pages": {"url": pages, "status_matches_committed_snapshot": True},
        "paper_sha256": digest(Path(f"output/pdf/novelty-es-{slug}.pdf")),
        "scope": "Large-file server SHA-256 metadata; direct downloaded-byte hashes for small files; no model inference.",
    }
    suffix = f"-{slug}" if continued else ""
    Path(f"reports/publication{suffix}.json").write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    Path(f"reports/publication-audit{suffix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "repositories"}, indent=2))


if __name__ == "__main__":
    main()
