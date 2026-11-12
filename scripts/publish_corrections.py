"""Publish dated documentation only; verify evidence and narrow Hub storage metadata.

No training/checkpoint/export writer is imported. Hub commit API and optimistic
parent guard: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.create_commit .
Use --publish only after reviewing staged artifacts; the default is a read-only plan.
"""

import argparse
import hashlib
import json
import posixpath
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv


def file_signature(info):
    """Use Git blob IDs plus server LFS SHA-256 and sizes for unchanged-file proof."""
    return {file.rfilename: dict(bytes=file.size, blob_id=file.blob_id,
                                sha256=file.lfs.sha256 if file.lfs else None) for file in info.siblings}


def portable_report(text):
    """Resolve repository-relative report links before moving the report to the Hub."""
    base = 'https://github.com/BurnyCoder/evolutionary-strategies-novelty-search-llm-training/blob/main/'

    def replace(match):
        target = match.group(1)
        if target.startswith(('https://', 'http://', '#')):
            return match.group(0)
        path, separator, fragment = target.partition('#')
        resolved = posixpath.normpath(posixpath.join('reports', path))
        if resolved.startswith('../') or resolved.startswith('/'):
            raise ValueError(f'Report link escapes the repository: {target}')
        return '](' + base + quote(resolved, safe='/') + separator + fragment + ')'

    return re.sub(r'\]\(([^)]+)\)', replace, text)


def check_storage_rule(before, after):
    """Accept only the Hub's observed LFS rule for this newly uploaded paper."""
    rule = 'corrections/2026-11-12/paper.pdf filter=lfs diff=lfs merge=lfs -text\n'
    if after != before + ('' if before.endswith('\n') else '\n') + rule:
        raise ValueError('Unexpected change to Hub storage rules')
    return rule.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, help='Use a short writable path on Windows for revision verification downloads.')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    load_dotenv()
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    api = HfApi()
    hub_report = args.release_dir / 'huggingface/correction-report.md'
    hub_report.parent.mkdir(parents=True, exist_ok=True)
    hub_report.write_text(portable_report((args.release_dir / 'reports/independent-corrections-2026-11-12.md').read_text(encoding='utf-8')), encoding='utf-8')
    receipt = dict(date=datetime.now(timezone.utc).isoformat(), mode='publish' if args.publish else 'read-only plan',
                   changed_scope='Documentation only; predecessor evidence, checkpoints and weights are preserved.', repositories=[])
    for slug, original_file in (('pilot', 'publication.json'),
                                ('continuation-seed0-gen100', 'publication-continuation-seed0-gen100.json')):
        original = json.loads((Path('reports') / original_file).read_text())
        release = args.release_dir / 'releases' / slug
        for role, url in {**original['models'], 'artifacts': original['artifacts']}.items():
            kind = 'dataset' if role == 'artifacts' else 'model'
            repo_id = url.split('/datasets/')[-1] if kind == 'dataset' else url.split('huggingface.co/')[-1]
            predecessor = original['revisions'][role]
            info = api.repo_info(repo_id, repo_type=kind, revision=predecessor, files_metadata=True)
            current = api.repo_info(repo_id, repo_type=kind, files_metadata=True)
            before = file_signature(info)
            card = release / ('README.md' if kind == 'dataset' else f'{role}/model/README.md')
            files = {
                'README.md': card,
                'corrections/2026-11-12/summary.json': release / 'summary.json',
                'corrections/2026-11-12/correction-report.md': hub_report,
                'corrections/2026-11-12/paper.pdf': args.release_dir / f'output/pdf/novelty-es-{slug}.pdf',
            }
            # This fixed allowlist cannot contain model weights, checkpoints or raw outputs.
            assert all(path.suffix in ('.md', '.json', '.pdf') and path.is_file() for path in files.values())
            entry = dict(repo_id=repo_id, repo_type=kind, predecessor=predecessor,
                         predecessor_created_at=api.list_repo_commits(repo_id, repo_type=kind, revision=predecessor)[0].created_at.isoformat(),
                         files={name: dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                for name, path in files.items()})
            if args.publish:
                # A retry may find the exact documentation commit already present.
                # Never overwrite a changed head: all expected bytes and unchanged
                # evidence below must match before adopting an existing revision.
                if current.sha == predecessor:
                    committed = api.create_commit(repo_id, repo_type=kind, parent_commit=current.sha,
                        operations=[CommitOperationAdd(name, path) for name, path in files.items()],
                        commit_message='Correct split labels, provenance and evidence limits in dated documentation revision')
                    revision, commit_url = committed.oid, committed.commit_url
                else:
                    revision = current.sha
                    prefix = 'datasets/' if kind == 'dataset' else ''
                    commit_url = f'https://huggingface.co/{prefix}{repo_id}/commit/{revision}'
                after_info = api.repo_info(repo_id, repo_type=kind, revision=revision, files_metadata=True)
                after = file_signature(after_info)
                assert set(after) == set(before) | set(files)
                metadata = {}
                if after.get('.gitattributes') != before.get('.gitattributes'):
                    old = Path(hf_hub_download(repo_id, '.gitattributes', repo_type=kind, revision=predecessor, cache_dir=args.cache_dir)).read_text()
                    new = Path(hf_hub_download(repo_id, '.gitattributes', repo_type=kind, revision=revision, cache_dir=args.cache_dir)).read_text()
                    metadata['.gitattributes'] = dict(added_rule=check_storage_rule(old, new), before=before['.gitattributes'], after=after['.gitattributes'])
                unchanged = {name: signature for name, signature in before.items() if name != 'README.md' and name not in metadata}
                assert all(after[name] == signature for name, signature in unchanged.items())
                for name, expected in entry['files'].items():
                    downloaded = Path(hf_hub_download(repo_id, name, repo_type=kind, revision=revision, cache_dir=args.cache_dir))
                    assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == expected['sha256']
                entry.update(revision=revision, commit_url=commit_url,
                             unchanged_predecessor_files=unchanged, storage_metadata=metadata, remote_hashes_match=True)
            elif current.sha != predecessor:
                entry['current_head'] = current.sha
                entry['note'] = 'Head differs from predecessor; publication will adopt it only if every planned byte and evidence invariant matches.'
            receipt['repositories'].append(entry)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({key: value for key, value in entry.items() if key not in ('files', 'unchanged_predecessor_files')}), flush=True)


if __name__ == '__main__':
    main()
