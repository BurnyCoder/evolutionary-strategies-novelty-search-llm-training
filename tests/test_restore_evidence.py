"""Restoration routes pinned evidence and separate weight exports to verifier inputs."""

import importlib.util
from pathlib import Path

from novelty_es.verification import inventory


def test_restoration_includes_both_evidence_sets_and_four_model_exports(tmp_path):
    spec = importlib.util.spec_from_file_location("restore_evidence", Path("scripts/restore_evidence.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.restoration_plan(tmp_path)
    assert len(plan) == 6
    for slug in ("pilot", "continuation"):
        items = [item for item in plan if item["local_dir"].is_relative_to(tmp_path / slug)]
        assert {item["local_dir"].relative_to(tmp_path / slug).as_posix() for item in items} == {
            ".", "main/model", "reward_only/model"}
        assert all(len(item["revision"]) == 40 for item in items)
        assert sum(item["repo_type"] == "dataset" for item in items) == 1


def test_live_service_bookkeeping_does_not_hide_evidence_changes(tmp_path):
    (tmp_path / "events.jsonl").write_text("original response")
    (tmp_path / "dashboard-error.log").write_text("request")
    before = inventory(tmp_path)
    (tmp_path / "dashboard-error.log").write_text("another request")
    assert inventory(tmp_path) == before
    (tmp_path / "events.jsonl").write_text("changed response")
    assert inventory(tmp_path) != before
