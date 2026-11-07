"""Read-only localhost dashboard with an explicit asset/API allowlist.

HTTP primitives: https://docs.python.org/3.12/library/http.server.html .
No arbitrary file serving, command execution, credentials, or remote mutation API.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def read_json(path: Path, fallback):
    """Atomic writers guarantee readers see either the previous or new complete JSON."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def dashboard_state(directory: Path) -> dict:
    """Expose compact experiment state and curves, not large logs/checkpoints or .env."""
    experiment = read_json(directory / "experiment.json", {"phase": "not_started"})
    return {
        "experiment": experiment,
        "summary": read_json(directory / "summary.json", {}),
        "progress": {name: read_json(directory / name / "progress.json", {}) for name in ("main", "reward_only")},
        "curves": {name: read_json(directory / name / "metrics.json", []) for name in ("main", "reward_only")},
    }


def serve(directory: Path, port: int) -> None:
    """Bind only localhost; static Pages output uses the same presentation template."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/status":
                body = json.dumps(dashboard_state(directory), allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
            elif route in ("/", "/index.html"):
                body = Path("web/index.html").read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"Live metrics: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
