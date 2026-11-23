import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
archive=Path("/data/chen/formal_execution/run_20261117_001732/five_data")
rows=[]
seen={}
for split, count in (("train",1200),("validation",400),("calibration",400),("test",400)):
    files=sorted((archive/split).glob("*.pkl"))
    assert [p.stem for p in files]==[f"{i:04}" for i in range(count)]
    for p in files:
        with p.open("rb") as f:
            digest=hashlib.file_digest(f,"sha256").hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate archive content: {p}, {seen[digest]}")
        seen[digest]=str(p)
        rows.append((split,p.stem,"offline/"+split,p.stat().st_size,digest))
target=root/"work/synthetic_split_manifest.csv"
with target.with_suffix(".tmp").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["split","mission","generator_namespace","bytes","sha256"])
    w.writerows(rows)
target.with_suffix(".tmp").replace(target)
print(f"{len(rows)} unique serialized archives")
