"""Check release boundaries without network access or loading model tensors."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from novelty_es.publishing import publish


@pytest.mark.parametrize("continuation", [False, True])
def test_publication_pins_revisions_and_includes_resumable_evidence(tmp_path, monkeypatch, continuation):
    """A completed release includes checkpoint hashes but excludes incidental local secrets."""
    state = {"phase": "finished", "config": {"model_id": "base/model", "model_revision": "base-sha"}}
    if continuation:
        state["config"].update(continuation={}, publication_suffix="seed0-gen100", artifact_repo="longer-artifacts")
    (tmp_path / "experiment.json").write_text(json.dumps(state))
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "README.md").write_text("Artifact documentation")
    (tmp_path / ".env").write_text("DO_NOT_PUBLISH=synthetic-test-value")
    (tmp_path / "main/model").mkdir(parents=True)
    (tmp_path / "main/checkpoint.pt").write_bytes(b"synthetic-checkpoint")
    (tmp_path / "main/pending.pt").write_bytes(b"unfinished-candidate")
    license_path = tmp_path / "base-license"
    license_path.write_text("Test license")
    uploads = []

    class FakeApi:
        def whoami(self):
            return {"name": "researcher"}

        def create_collection(self, **kwargs):
            return SimpleNamespace(url="https://huggingface.co/collections/researcher/study", slug="researcher/study")

        def create_repo(self, *args, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)
            return SimpleNamespace(oid=f"commit-{len(uploads)}")

        def add_collection_item(self, *args, **kwargs):
            pass

    monkeypatch.setattr("novelty_es.publishing.HfApi", FakeApi)
    monkeypatch.setattr("novelty_es.publishing.hf_hub_download", lambda *args, **kwargs: str(license_path))
    receipt = publish(tmp_path)
    manifest = json.loads((tmp_path / "artifact-manifest.json").read_text())
    assert manifest["files"]["main/checkpoint.pt"]["sha256"] == hashlib.sha256(b"synthetic-checkpoint").hexdigest()
    assert ".env" not in manifest["files"] and "main/pending.pt" not in manifest["files"]
    assert "main/checkpoint.pt" in uploads[-1]["allow_patterns"]
    assert "artifact-manifest.json" in uploads[-1]["allow_patterns"]
    assert receipt["revisions"] == {"main": "commit-1", "artifacts": "commit-2"}
    assert json.loads((tmp_path / "published.json").read_text()) == receipt
    suffix = "seed0-gen100" if continuation else "seed0"
    assert uploads[0]["repo_id"] == f"researcher/novelty-es-qwen2.5-0.5b-main-{suffix}"
    assert uploads[1]["repo_id"] == ("researcher/longer-artifacts" if continuation else "researcher/novelty-es-pilot-artifacts")


def test_active_run_cannot_be_published(tmp_path):
    """Do not hash or upload files while the experiment is still replacing them."""
    (tmp_path / "experiment.json").write_text('{"phase": "main"}')
    with pytest.raises(RuntimeError, match="report before publishing"):
        publish(tmp_path)


def test_continuation_cannot_overwrite_original_hub_repositories(tmp_path):
    """Reject colliding destinations before any remote API call."""
    (tmp_path / "experiment.json").write_text(json.dumps({"phase": "finished", "config": {"continuation": {}}}))
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="distinct"):
        publish(tmp_path)
