"""Geometry of independent predictors under paired composition and detour stress."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import pickle
import numpy as np
import pandas as pd
import torch
from server_paths import RUN
from grudata import schedule
from routeclock.filters import PositionIMM
from center import mean_projection
from common import truth
from grutrain import GRID
from age_panel import angle_deg,PRIMARY_N
from raft_rebuild import Raft
from summarize import estimate

STRICT=RUN/'stress_strict'
VIEWS=('reference9','structured0','structured16','detour9')


@lru_cache(maxsize=1)
def model():
    torch.set_num_threads(1)
    return Raft()


def one(k):
    arm=[];phases=[]
    detours=json.loads((STRICT/f'{k}_detour9.json').read_text())['detours']
    for name in VIEWS:
        with (STRICT/f'{k}_{name}.pkl').open('rb') as f:w=pickle.load(f)
        history=[[] for _ in range(16)];imm=[PositionIMM() for _ in history]
        by=[[] for _ in history];ih=[];events=[];targets=[]
        det_users={d['user']:d for d in detours}
        for u,t in schedule(w,41000+k):
            o=w.observe(u,t)
            history[u].append(o);imm[u].observe(o)
            n=len(history[u])
            relevant=(u in det_users and name in ('reference9','detour9')
                and det_users[u]['onset_s']-24<=t<=det_users[u]['end_s']+12)
            if n not in PRIMARY_N and not relevant:continue
            ih.append(history[u].copy());events.append((u,t,n))
            targets.append(mean_projection()@(imm[u].weights[:,None]*imm[u].x).ravel())
        for start in range(0,len(ih),48):
            for ix,rp in enumerate(model().predict(ih[start:start+48]),start):
                u,t,n=events[ix];ip=targets[ix];by[u].append((t,ip,rp))
                if n in PRIMARY_N:
                    for age in (2,3,5):
                        if t+age>180:continue
                        y=truth(w,u,t+age)
                        arm.append(dict(k=k,view=name,u=u,n=n,age=age,
                            IMM=angle_deg(ip[2*age],y),RAFT=angle_deg(rp[2*age],y)))
        if name in ('reference9','detour9'):
            for item in detours:
                u=item['user'];on=item['onset_s'];track=by[u];ts=[p[0] for p in track]
                for offset in range(-12,24):
                    tq=on+offset;j=np.searchsorted(ts,tq,side='right')-1
                    if j<0:raise ValueError('missing prefix')
                    t,ip,rp=track[j];age=tq-t
                    if age>10:continue
                    y=truth(w,u,tq)
                    at=lambda p:np.array([np.interp(age,GRID,col) for col in p.T])
                    phases.append(dict(k=k,view=name,u=u,offset_s=offset,phase=('pre' if offset<0 else 'during' if offset<12 else 'post'),
                              age=age,IMM=angle_deg(at(ip),y),RAFT=angle_deg(at(rp),y)))
    return arm,phases


if __name__=='__main__':
    with ProcessPoolExecutor(6) as pool:items=list(pool.map(one,range(3000,3012)))
    out=RUN/'stress_results';out.mkdir(exist_ok=True)
    arms=pd.DataFrame(r for a,b in items for r in a)
    phases=pd.DataFrame(r for a,b in items for r in b)
    arms.to_csv(out/'composition.csv.gz',index=False);phases.to_csv(out/'deviation.csv.gz',index=False)
    av=arms.groupby(['k','view','n','age'])[['IMM','RAFT']].mean().groupby(level=[0,1]).mean()
    r=[]
    for name in VIEWS:
        a=av.xs(name,level=1)
        r.append(dict(comparison='IMM_minus_RAFT',condition=name,**estimate(a.IMM-a.RAFT)))
    for phase in ('pre','during','post'):
        p=phases[phases.phase==phase]
        left=p[p.view=='detour9'].set_index(['k','u','offset_s']).sort_index()
        right=p[p.view=='reference9'].set_index(['k','u','offset_s']).sort_index()
        assert left.index.equals(right.index) and np.array_equal(left.age,right.age)
        diff=(left[['IMM','RAFT']]-right[['IMM','RAFT']]).groupby(level=0).mean()
        if phase=='pre':assert diff.abs().to_numpy().max()<1e-9
        for name in diff:r.append(dict(comparison=name+'_detour_minus_reference',condition=phase,**estimate(diff[name])))
    pd.DataFrame(r).to_csv(out/'summary.csv',index=False)
    print(len(arms),len(phases),flush=True)
