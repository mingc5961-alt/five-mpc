import json
from pathlib import Path
import sys
from time import monotonic

from runtime.remaining_runs import Runner

method = sys.argv[1]
root = Path(__file__).resolve().parent
start = monotonic()
result = Runner().execute(dict(model=method, phase="confirm", world=0, condition="T", mcs_margin_db=0, tube="default"))
result["elapsed_s"] = monotonic() - start
path = root / "work" / "pilot" / f"{method.lower()}_0db_T.json"
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(result, default=lambda value: value.tolist()))
temp.replace(path)
print(result["G_mbps"], result.get("R"), result["elapsed_s"])
