"""Reconstruct the original ZIP and verify every SHA-256. Python 3, no dependencies.
Run in Windows/macOS/Linux: python merge_parts.py
Only verify downloads: python merge_parts.py --verify-only
"""
from pathlib import Path
import argparse,hashlib,json,os
p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent;m=json.loads((root/'parts_manifest.json').read_text());target=root/m['original_name'];tmp=target.with_suffix('.zip.merging')
if not args.verify_only and (target.exists() or tmp.exists()):raise SystemExit('Output already exists. Move it away before retrying.')
for part in m['parts']:
 path=root/part['name']
 if not path.is_file() or path.stat().st_size!=part['bytes']:raise SystemExit('Missing/incomplete part: '+part['name'])
h=hashlib.sha256();out=None
try:
 if not args.verify_only:out=tmp.open('xb')
 for part in m['parts']:
  ph=hashlib.sha256()
  with (root/part['name']).open('rb') as src:
   while b:=src.read(8*1024*1024):
    ph.update(b);h.update(b)
    if out:out.write(b)
  if ph.hexdigest()!=part['sha256']:raise RuntimeError('SHA-256 mismatch: '+part['name'])
  print('Verified:',part['name'],flush=True)
 if h.hexdigest()!=m['original_sha256']:raise RuntimeError('Combined SHA-256 mismatch')
 if out:out.close();out=None;os.replace(tmp,target)
 print('All checks passed.' if args.verify_only else 'Ready to unzip: '+str(target))
finally:
 if out:out.close()
