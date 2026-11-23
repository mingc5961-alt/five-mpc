"""Read saved node traces, recompute the physical ledger, and plot mechanisms."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from routeclock.config import dump_json
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run)
events=json.loads((root/'events.json').read_text(encoding='utf-8'));pkgs=json.loads((root/'predictions.json').read_text(encoding='utf-8'));metrics=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
bits=sum(e.get('delivered_bits',0) for e in events);service=metrics['accounted_s'];recomputed=bits/service/1e6
if not np.isclose(recomputed,metrics['net_mbps'],rtol=1e-9):raise AssertionError('Event bits do not match summary')
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for i in range(metrics['N']):
    obs=[e for e in events if e['type']=='observation' and e['uav']==i];pred=[x for x in pkgs if x['uav']==i]
    data=[e for e in events if e['type']=='data' and e['uav']==i and 'pair' in e]
    axes[0].plot([x['origin'] for x in pred],[x['progress_std'][-1] for x in pred],label=f'UAV {i}')
    axes[1].step([e['start'] for e in data],[e['deadline']-e['prediction_origin'] for e in data],where='post')
    axes[2].step([e['start'] for e in data],[np.prod(e['pair']) for e in data],where='post')
axes[0].set_ylabel('10s progress std (m)');axes[1].set_ylabel('Next interval (s)');axes[2].set_ylabel('Active elements');axes[2].set_xlabel('Mission time (s)');axes[0].legend();fig.tight_layout();fig.savefig(root/'mechanism.png',dpi=160)
dump_json(root/'ledger_recomputed.json',{'bits':bits,'net_mbps':recomputed,'matches_summary':True,'observation_events':sum(e['type']=='observation' for e in events),'prediction_packages':len(pkgs)})
