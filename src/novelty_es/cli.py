"""Small user-facing phase wrapper; implementation lives in focused modules."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Dispatch explicit phases; only run creates the experiment's immutable clock."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reproducible full-weight novelty ES pilot")
    parser.add_argument("command", choices=["prepare", "download", "run", "recover", "report", "dashboard", "publish"])
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--run-dir", type=Path, default=Path(os.environ.get("NOVELTY_RUN_DIR", "artifacts/pilot")))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "prepare":
        from .data import load_data, prepare_data

        if Path("data/manifest.json").exists():
            load_data(Path("data"))
            print("Registered dataset hashes verified; existing data preserved.")
        else:
            print(json.dumps(prepare_data(Path("data"), config["data_seed"]), indent=2))
    elif args.command == "download":
        from huggingface_hub import snapshot_download

        for prefix in ("model", "embedding"):
            path = snapshot_download(
                config[f"{prefix}_id"],
                revision=config[f"{prefix}_revision"],
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE*", "NOTICE*"],
            )
            print(f"Pinned {prefix} downloaded to {path}")
    elif args.command in ("run", "recover"):
        from filelock import FileLock

        from .pilot import run_pilot
        from .reporting import report

        args.run_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.run_dir / "writer.lock"), timeout=0):
            if args.command == "recover":
                from .pilot import _git
                from .restart import initialize_recovery

                if not (args.run_dir / "experiment.json").exists() and _git("status", "--porcelain"):
                    raise RuntimeError("Publish and commit recovery changes before model execution")
                initialize_recovery(config, args.run_dir, _git("rev-parse", "HEAD"))
            run_pilot(config, args.run_dir)
            report(args.run_dir)
    elif args.command == "report":
        from .reporting import report

        report(args.run_dir)
    elif args.command == "dashboard":
        from .dashboard import serve

        serve(args.run_dir, int(os.environ.get("NOVELTY_DASHBOARD_PORT", "8765")))
    elif args.command == "publish":
        from .publishing import publish

        publish(args.run_dir)


if __name__ == "__main__":
    main()
