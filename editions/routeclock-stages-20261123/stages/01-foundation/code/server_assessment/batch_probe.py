import json
import os
import sys
import time
from pathlib import Path

os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
ROOT = Path('/data/chen/routeclock_server')
OUT = Path('/data/chen/server_assessment')
os.environ['ROUTECLOCK_TABPFN_CHECKPOINT'] = str(ROOT/'weights/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt')
sys.path.insert(0, str(ROOT/'routeclock_five_20261115'))
import numpy as np
import torch
from routeclock.foundation import tabpfn_regressor

torch.set_num_threads(4)
rng = np.random.default_rng(7)
X = rng.normal(size=(7200, 63))
y = rng.normal(size=7200)
query = rng.normal(size=(2400, 63))
torch.cuda.reset_peak_memory_stats()
t = time.perf_counter()
m = tabpfn_regressor('cuda')
m.fit(X, y)
torch.cuda.synchronize()
fit_s = time.perf_counter()-t
t = time.perf_counter()
pred = m.predict(query)
torch.cuda.synchronize()
report = dict(purpose='synthetic resource probe, not accuracy evaluation', train_shape=list(X.shape), query_shape=list(query.shape), fit_s=fit_s, predict_s=time.perf_counter()-t, finite=bool(np.isfinite(pred).all()), peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
(OUT/'batch_probe.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
