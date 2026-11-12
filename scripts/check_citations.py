"""Inventory citation destinations and check reachable external and existing local files.

HTTP success establishes a destination, not evidentiary support; the claim ledger
records the separate source reading. Requests use Python's standard URL library:
https://docs.python.org/3.12/library/urllib.request.html .
"""

import argparse
import concurrent.futures
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def request(url):
    """Record failures honestly; do not treat rate limits as nonexistent sources."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ResearchCitationCheck/1.0'}), timeout=20) as response:
            return dict(status=response.status, destination=response.url)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return dict(status=getattr(error, 'code', 'unavailable'), error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--external', action='store_true')
    args = parser.parse_args()
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
    external, local = {}, []
    for name in paths:
        path = Path(name)
        if path.suffix not in ('.md', '.py', '.html') or name.startswith('tests/'):
            continue
        text = path.read_text(encoding='utf-8')
        for url in re.findall(r'https?://[^\s<>\]"\)]+', text):
            url = url.rstrip('.,;:').split('#')[0]
            if any(char in url for char in ("'", '{', '}')) or url.startswith(('http://127.', 'http://localhost')):
                continue
            external.setdefault(url, set()).add(name)
        if path.suffix == '.md':
            for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text):
                if target.startswith(('http', '#')):
                    continue
                destination = path.parent / urllib.parse.unquote(target.split('#')[0])
                local.append(dict(location=name, label=label, target=target, exists=destination.exists()))
    output = {'local': local, 'external': {url: {'locations': sorted(locations)} for url, locations in sorted(external.items())}}
    if args.external:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for url, result in zip(output['external'], pool.map(request, output['external']), strict=True):
                output['external'][url].update(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'missing_local': [row for row in local if not row['exists']],
                      'external_unavailable': {url: row for url, row in output['external'].items() if row.get('status', 200) != 200}}, indent=2))


if __name__ == '__main__':
    main()
