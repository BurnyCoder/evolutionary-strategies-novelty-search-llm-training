"""Moved Hub documentation must retain working repository evidence links."""

import importlib.util
from pathlib import Path

import pytest


def test_hub_report_resolves_relative_links_and_preserves_external_links():
    spec = importlib.util.spec_from_file_location('publish_corrections', Path('scripts/publish_corrections.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = '[source](../docs/literature.md#method) [evidence](claim-ledger.json) [paper](https://arxiv.org/abs/2107.03374) [local](#findings)'
    result = module.portable_report(source)
    base = 'https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training/blob/main/'
    assert f'[source]({base}docs/literature.md#method)' in result
    assert f'[evidence]({base}reports/claim-ledger.json)' in result
    assert '[paper](https://arxiv.org/abs/2107.03374)' in result
    assert '[local](#findings)' in result


def test_only_exact_new_pdf_storage_rule_is_allowed():
    spec = importlib.util.spec_from_file_location('publish_corrections', Path('scripts/publish_corrections.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = '*.safetensors filter=lfs diff=lfs merge=lfs -text\n'
    rule = 'corrections/2026-11-12/paper.pdf filter=lfs diff=lfs merge=lfs -text\n'
    assert module.check_storage_rule(before, before + rule) == rule.strip()
    with pytest.raises(ValueError, match='storage rules'):
        module.check_storage_rule(before, rule)
    with pytest.raises(ValueError, match='storage rules'):
        module.check_storage_rule(before, before + rule + '*.pt text\n')
