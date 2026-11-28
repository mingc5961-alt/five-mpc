"""Apply the continuous 10 m test to stored paired forecasts, preserving initial labels."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import pandas as pd
from server_paths import RUN
from worlds import load_world
from age_panel import Geometry


def one(k):
    w=load_world('rebuild_holdout',k)
    geo=[Geometry(w.times,w.positions[u],w.velocities[u]) for u in range(w.n)]
    panel=RUN/'panel'/f'{k}.csv.gz'
    a=pd.read_csv(panel)
    keys=['u','n','t','age']
    unit=a[keys].drop_duplicates().copy()
    unit['label_new']=[geo[r.u].label(r.t,r.t+r.age,w.detour_windows[r.u]) for r in unit.itertuples()]
    pairs=a.merge(unit,on=keys,validate='many_to_one',sort=False)
    if 'label_sampled5' not in pairs:pairs['label_sampled5']=pairs['label']
    changed=int((pairs.label_sampled5!=pairs.label_new).sum())
    for folder in ('panel','independent_panel'):
        f=RUN/folder/f'{k}.csv.gz'
        d=pairs.copy() if folder=='panel' else pd.read_csv(f).merge(unit,on=keys,validate='many_to_one',sort=False)
        if 'label_sampled5' not in d:d['label_sampled5']=d['label']
        d['label']=d.pop('label_new')
        tmp=f.with_suffix(f'.{os.getpid()}.tmp')
        d.to_csv(tmp,compression='gzip',index=False);tmp.replace(f)
    return dict(world=k,queries=len(pairs),changed=changed)


if __name__=='__main__':
    with ProcessPoolExecutor(8) as p:rows=list(p.map(one,range(4000,4024)))
    (RUN/'geometry_label_audit.json').write_text(json.dumps(rows,indent=2))
