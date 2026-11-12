"""Restore immutable evidence AND separate inference exports, without model execution.

Download API: https://huggingface.co/docs/huggingface_hub/guides/download .
Revisions come from original receipts; documentation corrections never rewrite them.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def restoration_plan(root: Path) -> list[dict]:
    """Return concrete pinned downloads, also usable by offline workflow tests."""
    plan = []
    for slug, receipt in (("pilot", "publication.json"),
                          ("continuation", "publication-continuation-seed0-gen100.json")):
        publication = json.loads((Path("reports") / receipt).read_text())
        plan.append(dict(repo_id=publication["artifacts"].split("/datasets/")[1], repo_type="dataset",
                         revision=publication["revisions"]["artifacts"], local_dir=root / slug))
        for condition, url in publication["models"].items():
            plan.append(dict(repo_id=url.split("huggingface.co/")[1], repo_type="model",
                             revision=publication["revisions"][condition], local_dir=root / slug / condition / "model"))
    return plan


def main():
    """Restore exactly the published files; no training/evaluation imports are needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    from huggingface_hub import snapshot_download

    for item in restoration_plan(args.output_root):
        print(json.dumps(item, default=str), flush=True)
        if not args.dry_run:
            snapshot_download(**item, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
