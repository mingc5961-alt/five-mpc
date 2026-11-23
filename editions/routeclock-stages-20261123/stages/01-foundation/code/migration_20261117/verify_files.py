"""Verify the extracted, still-paused migration snapshot using only stdlib."""
import concurrent.futures
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify(entry):
    path = ROOT / entry['path']
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return entry['path'] + ': missing or size mismatch'
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != entry['sha256']:
        return entry['path'] + ': SHA256 mismatch'


if __name__ == '__main__':
    manifest = json.loads((ROOT/'migration_20261117/files_manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        errors = [result for result in pool.map(verify, manifest['files']) if result]
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"Verified {len(manifest['files'])} files; {manifest['total_bytes']/2**30:.2f} GiB. All SHA256 checks passed.")
