"""Map every tracked/reviewable surface to explicit dispositions in the claim ledger.

This inventory is a coverage index, not an automatic semantic fact checker.
The manually reviewed claim groups and their evidence remain authoritative.
"""

import json
import subprocess
from pathlib import Path

from novelty_es.verification import read, sha256


def groups(name: str) -> tuple[str, list[str]]:
    """Classify files by their actual role; unknown surfaces require explicit review."""
    if name in ('README.md', 'AGENTS.md'):
        return 'overview or governing protocol; empirical claims qualified by ledger', ['C01', 'C06', 'C09', 'C13', 'C14', 'C17', 'C25', 'C27', 'C33', 'C45']
    if name in ('LICENSE', 'NOTICE'):
        return 'license and attribution checked', ['C34']
    if name.startswith('configs/') or 'preregistration' in name:
        return 'frozen protocol preserved; commitments are not observations', ['C03', 'C11', 'C27', 'C28', 'C33']
    if name.startswith('data/'):
        return 'independently checked dataset hashes, witnesses and separation', ['C10', 'C11']
    if name.startswith('tests/'):
        return 'synthetic software assertions; not research-model results', ['C16', 'C24', 'C26', 'C45', 'C46', 'C47']
    if name.startswith('src/') or name.startswith('scripts/'):
        return 'implementation reviewed; test and historical-helper limits apply', ['C01', 'C02', 'C03', 'C04', 'C05', 'C08', 'C09', 'C12', 'C15', 'C16', 'C24', 'C25', 'C26', 'C45', 'C46', 'C47', 'C49']
    if name.startswith('docs/site/') or name.startswith('web/'):
        return 'snapshot numbers checked; visual/browser disposition in ledger', ['C13', 'C14', 'C17', 'C22', 'C23', 'C47', 'C48']
    if name.startswith('docs/'):
        return 'method/source/operations review with explicit historical limits', ['C01', 'C04', 'C05', 'C06', 'C08', 'C09', 'C25', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C49']
    if name.startswith('reports/figures/') or name.startswith('output/pdf/'):
        return 'derived figures/papers checked against source summaries and rendered pages', ['C13', 'C14', 'C17', 'C20', 'C21', 'C22', 'C23', 'C51']
    if name.startswith('reports/'):
        if 'publication' in name:
            return 'immutable original receipt or separately dated correction receipt', ['C27', 'C49', 'C50']
        if 'audit' in name or name.endswith('source-conformance.json'):
            return 'historical assertions retained, not premises of independent checks', ['C06', 'C29', 'C30', 'C49']
        if 'independent' in name or 'embedding-inputs' in name or 'claim-' in name:
            return 'new bounded evidence/check record; inspect explicit gaps', ['C26', 'C27', 'C30', 'C48', 'C52']
        return 'experiment record or generated summary; repeated claims linked to primary evidence', ['C06', 'C13', 'C14', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C24', 'C28', 'C29', 'C31', 'C32', 'C49']
    if name.startswith('.github/') or name in ('.env.example', '.gitattributes', '.gitignore', '.python-version', 'pyproject.toml', 'uv.lock'):
        return 'configuration/build metadata; no capability claim', ['C45', 'C50']
    raise ValueError(f'Unclassified surface requires review: {name}')


def main():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
    ledger = read(Path('reports/claim-ledger.json'))
    ids = {row['id'] for row in ledger['claims']}
    entries = {}
    for name in sorted(set(names)):
        if name == 'reports/claim-inventory.json':
            continue  # Self-hashing would be recursive; this exclusion is explicit below.
        disposition, claims = groups(name)
        assert set(claims) <= ids
        entries[name] = dict(sha256=sha256(Path(name)), disposition=disposition, claim_ids=claims)
    verified = read(Path('reports/independent-verification-2026-11-12.json'))
    raw = {root: {name: {**value, 'claim_ids': ['C11'] if root in ('data', 'fresh') else
                         ['C49'] if 'audit' in name else ['C07', 'C09', 'C12', 'C13', 'C18', 'C20', 'C28', 'C30']}
                  for name, value in files.items()} for root, files in verified['input_files'].items()}
    result = dict(date=ledger['date'], scope='Grouped claim coverage, not proof of every conceivable assertion.',
                  excluded=['This inventory itself (recursive hash)', 'Ignored caches, credentials and live service bookkeeping'],
                  files=entries, raw_evidence=raw, unclassified_files=[])
    Path('reports/claim-inventory.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Assigned dispositions to {len(entries)} repository files and {sum(map(len, raw.values()))} evidence files.')


if __name__ == '__main__':
    main()
