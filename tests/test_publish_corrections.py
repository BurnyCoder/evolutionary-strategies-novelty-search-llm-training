"""Moved Hub documentation must retain working repository evidence links."""

import importlib.util
from pathlib import Path


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
