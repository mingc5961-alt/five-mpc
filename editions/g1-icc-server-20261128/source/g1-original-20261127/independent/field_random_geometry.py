"""Evaluate all random flights, including first passes, and audit persistent route repetition."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import gzip, csv, json, re, zlib

import numpy as np
import pandas as pd

from field_worlds import OUT
from history import put
from paired_grid import BASE
from policy_env import initialize
from predictors import raft_model
from paired_models import parts
from panel_types import angle_deg
from slots import compute_slot

N=(6,8,12,24,40);AGES=(2.,3.,5.)
MODELS=('adaptive','motion','Fstar','RAFT','Chronos')
def paid_prefix(t,xyz,seed):
    from routeclock.geometry import angles, unit, position_covariance, spherical_to_xyz
    from routeclock.world import Observation
    rng=np.random.default_rng(seed)
    times=np.cumsum(np.r_[.0031,rng.uniform(3.,4.2,int(t[-1]/3)+2)])
    index={i:[a for a in AGES if times[i-1]+a<=t[-1]]
           for i in N if i<=len(times) and times[i-1]+min(AGES)<=t[-1]}
    if not index:return [],{}
    sample=lambda sec:np.array([np.interp(sec,t,col) for col in xyz.T])
    std=np.array([1.,*np.deg2rad([.1,.1]),.2])
    obs=[];by={}
    for i,sec in enumerate(times[:max(index)],1):
        p=sample(sec);past=max(t[0],sec-.10)
        v=(p-sample(past))/(sec-past)
        z=np.r_[np.linalg.norm(p),angles(p),unit(p)@v]+std*rng.standard_normal(4)
        obs.append(Observation(sec,spherical_to_xyz(z),position_covariance(z,std),float(z[3]),std[3],z))
        if i in index:
            by[i]=[(a,'all',int(i==len(times) or times[i]>sec+a),sample(sec+a)) for a in index[i]]
    return obs,by

def recurrence(t,p):
    # Compare pairs one fixed distance apart along the route, allowing changes in flying speed.
    sample=np.arange(t[0],t[-1],.5)
    if len(sample)<8:return None
    path=np.column_stack([np.interp(sample,t,a) for a in p.T]);arc=np.r_[0.,np.linalg.norm(np.diff(path,axis=0),axis=1).cumsum()]
    span=np.linalg.norm(np.ptp(path,axis=0))
    if span<.2 or arc[-1]<1:return None
    keep=np.r_[True,np.diff(arc)>1e-8];grid=np.linspace(0.,arc[-1],257)
    shape=np.column_stack([np.interp(grid,arc[keep],col[keep]) for col in path.T])
    direction=np.gradient(shape,axis=0); direction/=np.maximum(1e-12,np.linalg.norm(direction,axis=1,keepdims=True))
    best=0.
    for lag in range(26,129):
        near=np.linalg.norm(shape[lag:]-shape[:-lag],axis=1)<=.05*span
        same=(direction[lag:]*direction[:-lag]).sum(axis=1)>=np.cos(np.pi/6)
        best=max(best,float((near&same).mean()))
    return best

def geometry_job(r):
    u=r['id'];source=r['source']
    with np.load(r['track']) as d:t,p=d['t'],d['p']
    score=recurrence(t,p)
    if r['label']!='random':return dict(source=source,flight=u,task=r['task'],label=r['label'],repeat_score=score)
    with compute_slot():
        if source=='amov':
            with gzip.open(BASE/'field/scores'/f'amov_{u}.csv.gz','rt') as fp:rows=list(csv.DictReader(fp))
            frame=pd.DataFrame(rows);frame=frame[frame.n.astype(int).isin(N)&frame.age.astype(float).isin(AGES)].copy()
            for k in MODELS:frame[k+'_deg']=frame[k+'_deg'].astype(float)
        else:
            import paid_full as pf
            p=p+np.array([300.,0.,0.]);seed=[271128,zlib.crc32(u.encode())]
            obs,by=paid_prefix(t,p,seed)
            if not by:return dict(source=source,flight=u,task=r['task'],label=r['label'],repeat_score=score,queries=0)
            net=pf.model();st=pf.State(pf.load_config(),net,'G');rows=[]
            for i,o in enumerate(obs,1):
                net.n=i;pack=st.update(o,net)['G']
                if i not in by:continue
                centers=[pf.forecast(pack,'G',1)[0],pack[0],parts(pack,'Fstar')[0],
                         raft_model().predict([obs[:i]])[0],pf.chrono().batch([obs[:i]])[0][0]]
                for a,label,natural,truth in by[i]:
                    rows.append(dict(n=i,age=a,natural=natural,**{k+'_deg':float(angle_deg(c[round(2*a)],truth)) for k,c in zip(MODELS,centers)}))
            frame=pd.DataFrame(rows)
            frame.to_csv(OUT/'open_scores'/f'neurobem_{u}.csv',index=False)
        if not len(frame):return dict(source=source,flight=u,task=r['task'],label=r['label'],repeat_score=score,queries=0)
        angles=frame.groupby(['n','age'])[[k+'_deg' for k in MODELS]].mean().mean()
        tag=re.search(r'random points\s+(\d+)',r['task'],re.I) if source=='neurobem' else None
        return dict(source=source,flight=u,task=r['task'],label=r['label'],program=tag.group(1) if tag else u,
                    repeat_score=score,queries=len(frame),errors={k:float(angles[k+'_deg']) for k in MODELS})
def main():
    initialize();(OUT/'open_scores').mkdir(exist_ok=True)
    tracks=[r for s in ('amov','neurobem') for r in json.loads((OUT/(s+'_inputs.json')).read_bytes())['tracks']]
    with ProcessPoolExecutor(12) as pool:rows=list(pool.map(geometry_job,tracks))
    reports={}
    for src in ('amov','neurobem'):
        selected=[r for r in rows if r['source']==src and r['label']=='random']
        for label,flights in [('all',selected),('low_repetition',[r for r in selected if r['repeat_score'] is not None and r['repeat_score']<.7])]:
            valid=[r for r in flights if r.get('queries',0)]
            grouped=defaultdict(list)
            for r in valid:grouped[r['program']].append([r['errors'][m] for m in MODELS])
            if not grouped:continue
            means=np.array([np.mean(a,axis=0) for a in grouped.values()]);delta=means[:,1:]-means[:,0,None]
            ix=np.random.default_rng(271128).integers(len(means),size=(20000,len(means)))
            reports[src+'_'+label]=dict(flights=len(valid),programs=len(means),queries=sum(r['queries'] for r in valid),
                angle_deg=dict(zip(MODELS,means.mean(axis=0))),baseline_minus_adaptive_ci95=dict(zip(MODELS[1:],np.quantile(delta[ix].mean(1),[.025,.975],axis=0).T.tolist())))
    put(OUT/'recurrence_and_angles.json',dict(rule='maximum same-direction near-position match across constant arc-lags from 10% to 50% of total path; 5% bounding span and 30 degrees; strong repetition >=0.70; diagnostic, not a stochasticity test',
             summary=reports,tracks=rows))
if __name__=='__main__':main()
