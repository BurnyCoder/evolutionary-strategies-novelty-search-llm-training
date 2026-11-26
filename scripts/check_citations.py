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


def fetch(url):
    """Return HTTP status and destination without collapsing unavailable links into missing ones."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ResearchCitationCheck/1.0'}), timeout=20) as response:
            return dict(status=response.status, destination=response.url)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return dict(status=getattr(error, 'code', 'unavailable'), error=str(error))


def request(url):
    """A GitHub blob's raw representation can verify its contents during a web interface outage."""
    result = fetch(url)
    if result['status'] in (429, 500, 502, 503, 504, 'unavailable'):
        blob = re.fullmatch(r'https://github\.com/([^/]+/[^/]+)/blob/(.+)', url)
        if blob:
            result['raw_representation'] = fetch(f'https://raw.githubusercontent.com/{blob[1]}/{blob[2]}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--tracked-only', action='store_true', help='Check the published tree, including test sources.')
    args = parser.parse_args()
    options = [] if args.tracked_only else ['--others', '--exclude-standard']
    paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', *options]).decode().split('\0')
    external, local, fixtures = {}, [], []
    for name in paths:
        path = Path(name)
        if path.suffix not in ('.md', '.py', '.html') or (name.startswith('tests/') and not args.tracked_only):
            continue
        text = path.read_text(encoding='utf-8')
        for url in re.findall(r'https?://[^\s<>\]"\)]+', text):
            url = url.rstrip('.,;:').split('#')[0]
            if any(char in url for char in ("'", '{', '}', '\\')) or url.startswith(('http://127.', 'http://localhost')):
                continue
            if name == 'tests/test_publishing.py' and url == 'https://huggingface.co/collections/researcher/study':
                fixtures.append(dict(location=name, url=url, purpose='URL returned by a publishing test double'))
                continue
            external.setdefault(url, set()).add(name)
        if path.suffix == '.md':
            for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text):
                if target.startswith('http'):
                    continue
                relative, _, fragment = urllib.parse.unquote(target).partition('#')
                destination = path.parent / relative if relative else path
                exists = destination.exists()
                if exists and fragment and destination.suffix == '.md':
                    # The linked headings in this project use GitHub's ordinary lowercase, hyphenated slugs.
                    # https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#section-links
                    headings = re.findall(r'^#{1,6}\s+(.+)$', destination.read_text(encoding='utf-8'), re.M)
                    exists = fragment in {re.sub(r'[^\w -]', '', h.lower()).replace(' ', '-') for h in headings}
                local.append(dict(location=name, label=label, target=target, exists=exists))
    output = {'local': local, 'test_fixtures': fixtures,
              'external': {url: {'locations': sorted(locations)} for url, locations in sorted(external.items())}}
    if args.external:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for url, result in zip(output['external'], pool.map(request, output['external']), strict=True):
                output['external'][url].update(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'missing_local': [row for row in local if not row['exists']],
                      'external_unavailable': {url: row for url, row in output['external'].items()
                                               if row.get('status', 200) != 200 and row.get('raw_representation', {}).get('status') != 200}}, indent=2))


if __name__ == '__main__':
    main()
