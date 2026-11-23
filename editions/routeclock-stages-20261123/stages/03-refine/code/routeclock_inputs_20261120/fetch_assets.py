"""Restore pinned experiment assets using sources.tsv and SHA256SUMS."""
import csv
import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import requests

root = Path(__file__).resolve().parent
expected = {}
for line in (root / 'SHA256SUMS').read_text().splitlines():
    key, relative = line.split('  ', 1)
    expected[relative] = key
urls = defaultdict(list)
with (root / 'sources.tsv').open() as source:
    for row in csv.DictReader(source, delimiter='\t'):
        urls[row['local_path']].append(row['source_url'])

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def restore(rel, candidates):
    dest = root / rel
    if dest.is_file() and digest(dest) == expected[rel]:
        print('verified', rel, flush=True)
        return
    # The dated PX4 index and retained API response are snapshots of mutable metadata.
    if rel in ('px4/dbinfo.json.gz', 'cmu/figshare-v1.json'):
        print('use retained metadata snapshot', rel, flush=True)
        return
    if dest.parent.is_symlink():
        raise RuntimeError(f'Restore {rel} in a separate input directory to preserve the linked archive')
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in candidates:
        for environment_proxy in (False, True):
            try:
                with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                    pending = Path(tmp.name)
                    with requests.Session() as session:
                        session.trust_env = environment_proxy
                        with session.get(url, stream=True, timeout=(10, 90)) as response:
                            response.raise_for_status()
                            for chunk in response.iter_content(1024 * 1024):
                                tmp.write(chunk)
                if digest(pending) != expected[rel]:
                    raise ValueError('checksum mismatch')
                os.replace(pending, dest)
                print('downloaded', rel, flush=True)
                return
            except (OSError, requests.RequestException, ValueError) as error:
                print(rel, type(error).__name__, flush=True)
            finally:
                if 'pending' in locals():
                    pending.unlink(missing_ok=True)
    raise RuntimeError(f'No verified download: {rel}')

if __name__ == '__main__':
    for path, candidates in urls.items():
        restore(path, candidates)
