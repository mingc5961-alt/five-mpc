"""Check published cells against the frozen design and recover integer risk counts."""
import csv
import hashlib
import json
from pathlib import Path
import math

root = Path(__file__).resolve().parent
manifest = json.loads((root/"baseline_manifest.json").read_text())
rows = []
for file in sorted((root/"results/baselines").glob("*/complete.json")):
    record = json.loads(file.read_text())
    cfg = record["frozen_config"]
    r = record["result"]
    key = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    assert key == record["config_sha256"] and cfg["inputs"] == manifest["inputs"], file
    assert cfg["namespace"] == manifest["namespace"] and cfg["world"] in manifest["worlds"], file
    assert all(cfg[k] == r[k] for k in ("model","condition","mcs_margin_db","world")), file
    assert file.parent.name == f"{r['world']:03}_{r['condition']}_{r['model']}" and r["mcs_margin_db"] == 0, file
    assert r["model"] in manifest["methods"] and r["condition"] in manifest["budgets"], file
    assert len(r["bits"]) == 16 and r["probes_per_user"] == 9000, file
    assert abs(sum(r["ledger_s"])-180)<1e-6 and abs(sum(r["bits"])/180e6-r["G_mbps"])<1e-7,file
    n = len(r["bits"]) * r["probes_per_user"]
    total = round(n*r["R"])
    u = round(n*r["unavailable_fraction"])
    b = total-u
    assert abs(total-n*r["R"])<1e-7 and abs(u-n*r["unavailable_fraction"])<1e-7 and 0<=u<=total<=n,file
    rows.append(dict(world=r["world"],budget=r["condition"],method=r["model"],N=n,U=u,B=b,
                     risk_pct=100*total/n,geom_given_action_pct=100*b/(n-u) if n>u else None,G_mbps=r["G_mbps"]))
if rows:
    target=root/"work/baseline_audit.csv"
    temp=target.with_suffix(".tmp")
    with temp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(target)
print(len(rows),"completed cells passed")
