"""Reproduce three archived full-service runs without writing historical caches."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from .data import ROOT,BASE

arc=Path('/data/chen/routeclock_archives/paired600/runs/paired600')
root=BASE/'reproduction';root.mkdir(exist_ok=True)
reports={}
for method in ['F','I','RC']:
    original=arc/'test/cost3.1_budget5'/f'lifecycle_frozen_deadline_value_{method}_00'
    dest=root/method
    if not (dest/'metrics.json').exists():
        cmd=[sys.executable,str(ROOT/'tools/ch6_experiment.py'),'network','--out',str(dest),
             '--split','test','--cost-ms','3.1','--budget-pct','5','--method',method,
             '--use-calibration','deadline','--bundle',str(arc/'models/frozen/bundle.pkl')]
        if method=='F':cmd+=['--winner',str(arc/'fixed/cost3.1_budget5/winner.json')]
        subprocess.run(cmd,check=True)
    current=json.loads((dest/'metrics.json').read_text());previous=json.loads((original/'metrics.json').read_text())
    keys=['net_mbps','uav_mbps','sensing_s','data_s','unavailable_s']
    for k in keys:np.testing.assert_allclose(current[k],previous[k],atol=1e-6,rtol=1e-9)
    reports[method]=dict(reference=str(original),actual_mbps=current['net_mbps'],archive_mbps=previous['net_mbps'],verified=keys)
    (root/'summary.json').write_text(json.dumps(reports,indent=2))
