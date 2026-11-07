"""Durable local artifacts and UTC event logging shared by every experiment phase.

Atomic file replacement: https://docs.python.org/3.12/library/os.html#os.replace
RNG limitations: https://docs.pytorch.org/docs/stable/notes/randomness.html
Only load checkpoints produced locally by this project, never arbitrary pickle files.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


class BudgetExpired(RuntimeError):
    """A persistent wall-clock allowance was exhausted, including downtime."""


def check_budget(deadline: float, now: float | None = None) -> None:
    """Check absolute UTC seconds instead of resetting a duration on resume."""
    if (time.time() if now is None else now) >= deadline:
        raise BudgetExpired("The registered absolute phase deadline has passed")


def utc_now() -> str:
    """Return an unambiguous, millisecond-resolution timestamp for every event."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _replace(temporary: Path, destination: Path) -> None:
    """Retry a short Windows reader lock while retaining atomic replacement.

    Windows rename requires compatible open-handle sharing modes:
    https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew
    A bounded retry preserves the old file and never resets an experiment clock.
    """
    for attempt in range(20):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 19:
                raise
            time.sleep(0.05)


def atomic_json(path: Path, value: dict | list) -> None:
    """Flush data before replacing a same-directory file, preserving the old on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace(temporary, path)


def _rng_state() -> dict:
    """Convert NumPy's state to safe primitive/tensor types for weights-only loading."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def save_checkpoint(path: Path, state: dict) -> None:
    """Save complete resumable state atomically; keep the previous valid file on error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save({**state, "rng_state": _rng_state()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(temporary, path)


def load_checkpoint(path: Path, restore_rng: bool = False) -> dict:
    """Load primitive/tensor state and optionally restore all random streams exactly."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if restore_rng:
        rng = state["rng_state"]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        numpy_state = rng["numpy"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    return state


class EventLog:
    """Append full JSON events to disk and stdout; no prompt/answer truncation."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1, newline="\n")

    def emit(self, event: str, **fields) -> None:
        """Flush each event so a crash leaves a readable real-time record."""
        line = json.dumps({"timestamp": utc_now(), "event": event, **fields}, ensure_ascii=False, allow_nan=False)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self) -> None:
        """Release the stream after the phase's final event is durable."""
        self.stream.close()


def read_jsonl(path: Path) -> list[dict]:
    """Allow only an unfinished final line after a crash; reject corruption in the middle."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return result


def configure_runtime(seed: int) -> None:
    """Pin streams and request deterministic CUDA algorithms within this environment."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
