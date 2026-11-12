"""Small user-facing phase wrapper; implementation lives in focused modules."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Dispatch explicit phases; run/recover create their registered clocks."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reproducible full-weight novelty ES pilot")
    parser.add_argument("command", choices=["prepare", "download", "run", "recover", "report", "report-all", "verify", "dashboard", "publish"])
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--run-dir", type=Path, default=Path(os.environ.get("NOVELTY_RUN_DIR", "artifacts/pilot")))
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--continuation-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--fresh-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.command in ("verify", "report-all"):
        required = ["pilot_dir", "continuation_dir", "output_root"]
        if args.command == "verify":
            required += ["data_dir", "fresh_dir"]
        if any(getattr(args, name) is None for name in required):
            parser.error("Explicit paths required: " + ", ".join("--" + name.replace("_", "-") for name in required))
        from .verification import separate_output

        separate_output(args.output_root, [args.pilot_dir, args.continuation_dir])
        if args.command == "verify":
            from .verification import verify_all

            verify_all(args.pilot_dir, args.continuation_dir, args.data_dir, args.fresh_dir, args.output_root)
        else:
            import shutil

            from .reporting import refresh_corrections, report

            args.output_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile("README.md", args.output_root / "README.md")
            report(args.pilot_dir, args.output_root, staged=True)
            report(args.continuation_dir, args.output_root, staged=True)
            refresh_corrections(args.output_root)
        return
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

        report(args.run_dir, args.output_root or Path("."), staged=args.output_root is not None)
    elif args.command == "dashboard":
        from .dashboard import serve

        serve(args.run_dir, int(os.environ.get("NOVELTY_DASHBOARD_PORT", "8765")))
    elif args.command == "publish":
        from .publishing import publish

        publish(args.run_dir)


if __name__ == "__main__":
    main()
