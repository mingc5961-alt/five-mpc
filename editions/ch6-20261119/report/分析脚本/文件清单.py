"""Index archived files and verify the two published scientific manifests."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
primary = (root/'主实验归档').resolve()
extra = (root/'追加实验归档').resolve()
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

checks = {}
for line in (primary/'SHA256SUMS').read_text().splitlines():
    h, name = line.split('  ', 1)
    checks[primary/name] = h
for name,h in json.loads((extra/'sha256.json').read_text()).items():
    checks[extra/name] = h
rows = []
for base in (primary,extra):
    for p in sorted(base.rglob('*')):
        if not p.is_file(): continue
        h = digest(p)
        if p in checks: assert checks[p] == h,p
        rows.append((base.name, str(p.relative_to(base)),p.stat().st_size,h,int(p in checks),str(p)))
assert set(checks) <= {Path(r[-1]) for r in rows}
with (root/'数据表/归档文件逐项索引.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    w=csv.writer(stream)
    w.writerow(('归档','相对路径','字节','SHA256','原校验清单收录','chen路径'))
    w.writerows(rows)
print(len(rows),len(checks))
