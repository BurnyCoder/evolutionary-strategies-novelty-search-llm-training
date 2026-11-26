"""Verify public GitHub and Pages bytes and the six pinned Hub documentation revisions."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from huggingface_hub import HfApi

from novelty_es.verification import read, sha256

REPOSITORY = "BurnyCoder/evolutionary-strategies-novelty-search-llm-training"
PAGES = "https://burnycoder.github.io/evolutionary-strategies-novelty-search-llm-training/"


def check_file(item):
    """A successful HTTP response alone cannot establish that the revised bytes are live."""
    path, url = item
    with urlopen(Request(url, headers={"User-Agent": "ResearchReleaseVerification/1.0"}), timeout=40) as response:
        body = response.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != sha256(path):
        raise ValueError(f"Remote SHA-256 differs from the staged file: {url}")
    return dict(url=url, bytes=len(body), sha256=digest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-dir", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--hub-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", a.revision):
        raise ValueError("Supply the immutable GitHub merge SHA")
    root = a.release_dir
    paths = [root / "README.md"]
    for folder in ("reports", "docs/site", "output/pdf"):
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file() and p.name != ".nojekyll")
    jobs = []
    for path in sorted(paths):
        relative = quote(path.relative_to(root).as_posix(), safe="/")
        jobs.append((path, f"https://raw.githubusercontent.com/{REPOSITORY}/{a.revision}/{relative}"))
        if relative.startswith("docs/site/"):
            jobs.append((path, PAGES + relative.removeprefix("docs/site/") + "?revision=" + a.revision))
    with ThreadPoolExecutor(max_workers=5) as pool:
        verified = list(pool.map(check_file, jobs))
    receipt = read(a.hub_receipt)
    api = HfApi()
    collection = read(Path("reports/publication.json"))["collection"]
    members = {(i.item_id, i.item_type) for i in api.get_collection(collection.split("/collections/")[-1]).items}
    repos = receipt["repositories"]
    expected = {(r["repo_id"], r["repo_type"]) for r in repos}
    if len(repos) != 6 or len(expected) != 6 or not expected <= members:
        raise ValueError("The collection does not contain all six releases")
    for repo in repos:
        if (
            not repo["remote_hashes_match"]
            or api.repo_info(repo["repo_id"], repo_type=repo["repo_type"]).sha != repo["revision"]
        ):
            raise ValueError(f"Unverified or superseded Hub correction: {repo['repo_id']}")
    result = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        github_revision=a.revision,
        files=verified,
        collection=collection,
        hub_revisions={r["repo_id"]: r["revision"] for r in repos},
    )
    a.output.parent.mkdir(exist_ok=True, parents=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {len(verified)} GitHub/Pages objects and all six current Hub revisions.")


if __name__ == "__main__":
    main()
