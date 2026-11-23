"""CMU same-target E30, preserving sampled route memory across earlier flights."""
import argparse
import copy
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
from routeclock.config import load_config
from routeclock.geometry import angles,angle_error
from routeclock.predictors import Tracker,apply_gate
from routeclock.world import Observation
from .data import BASE,ASSETS
from .real import local_enu

def observations(t,xyz):
    # Real observation at or immediately before each 3 s query, never forward-look.
    ix=np.unique(t.searchsorted(np.arange(t[0],t[-1],3.),side='right')-1)
    for j in ix:
        yield Observation(float(t[j]-t[0]),xyz[j],np.eye(3),0.,1e6,np.zeros(4))

def import_maps(memory,maps):
    for fragments,connections in maps:
        offset=len(memory.fragments)
        memory.fragments.extend(copy.deepcopy(fragments))
        memory.connections.update({(a+offset,b+offset):v for (a,b),v in connections.items()})

def run(phase):
    cfg=load_config()
    archived='/data/chen/routeclock_archives/paired600/runs/paired600/models/frozen/bundle.pkl'
    with open(archived,'rb') as f:gate=pickle.load(f)['gates']['RC']
    meta=pd.read_csv(ASSETS/'cmu/parameters.csv').sort_values(['date','local_time','flight'])
    meta=meta[meta.route.str.startswith('R')]
    dates=sorted(meta.date.unique());dev=set(dates[:int(np.ceil(.6*len(dates)))])
    maps={};prior={};out=BASE/'cmu_history_dose'/phase;out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ASSETS/'cmu/flights.zip') as archive:
        def read(row):
            with archive.open(f'flights/{row.flight}.csv') as f:return pd.read_csv(f)
        axes=['position_x','position_y','position_z']
        origin=read(next(meta.itertuples(index=False)))[axes].iloc[0].to_numpy()
        for row in meta.itertuples(index=False):
            d=read(row);t=d.time.to_numpy();llh=d[axes].to_numpy()
            if len(t)<2 or not np.isfinite(llh).all() or (np.diff(t)<=0).any():continue
            if np.quantile(np.diff(t),.95)>.25 or np.max(np.diff(t))>.5:continue
            xyz=local_enu(llh,origin);obs=list(observations(t,xyz))
            key=row.route
            ids=prior.setdefault(key,[])
            target=(row.date in dev)==(phase=='development')
            dest=out/f'{row.flight:03}.csv'
            if target and not dest.exists():
                metrics=[]
                for k in [0,1,3,5]:
                    if len(ids)<k:continue
                    used=ids[-k:] if k else []
                    tracker=Tracker(cfg);inherited=False
                    for o in obs:
                        tracker.observe(o)
                        if used and not inherited and tracker.memory.active is not None:
                            import_maps(tracker.memory,[maps[i] for i in used]);inherited=True
                        pkg=tracker.issue()
                        at=t[0]+o.t+5
                        if at>t[-1] or o.position[2]-xyz[0,2]<5:continue
                        j=np.searchsorted(t,at)
                        if t[j]-t[j-1]>.25:continue
                        truth=np.array([np.interp(at,t,axis) for axis in xyz.T])
                        if truth[2]-xyz[0,2]<5:continue
                        apply_gate(pkg,gate,pkg.features,'RC','P')
                        for m in ['I','P','RC']:
                            residual=angle_error(angles(pkg.at(m,5)),angles(truth))
                            metrics.append(dict(flight=row.flight,date=row.date,route=key,K=k,history_ids=' '.join(map(str,used)),
                                      model=m,origin_s=o.t,error_deg=np.linalg.norm(np.rad2deg(residual))))
                tmp=dest.with_suffix('.part')
                pd.DataFrame(metrics,columns=['flight','date','route','K','history_ids','model','origin_s','error_deg']).to_csv(tmp,index=False);tmp.replace(dest)
            tracker=Tracker(cfg)
            for o in obs:tracker.observe(o)
            if tracker.memory.fragments:
                maps[row.flight]=(tracker.memory.fragments,tracker.memory.connections)
                ids.append(row.flight)
    frames=[pd.read_csv(f) for f in sorted(out.glob('[0-9]*.csv'))]
    d=pd.concat(frames,ignore_index=True)
    means=d.groupby(['flight','date','route','K','model'],as_index=False).agg(mean_angle_error_deg=('error_deg','mean'),origins=('error_deg','count'))
    means.to_csv(out/'per_flight.csv',index=False)
    common=means.groupby('flight').K.max();common=common[common==5].index
    means[means.flight.isin(common)].groupby(['K','model']).agg(mean_deg=('mean_angle_error_deg','mean'),flights=('flight','nunique')).to_csv(out/'dose_common.csv')
    wide=means.pivot(index=['flight','date','route'],columns=['K','model'],values='mean_angle_error_deg')
    diff=pd.concat([(wide[3,'RC']-wide[3,'I']).rename('RC3_minus_I3_deg'),
                    (wide[3,'RC']-wide[0,'RC']).rename('RC3_minus_RC0_deg')],axis=1).dropna().reset_index()
    diff.to_csv(out/'paired.csv',index=False)
    groups=list(diff.groupby('date'));rng=np.random.default_rng(271828);summary={}
    for col in ['RC3_minus_I3_deg','RC3_minus_RC0_deg']:
        groupsums=np.array([[v[col].sum(),len(v)] for _,v in groups])
        boots=groupsums[rng.integers(len(groups),size=(5000,len(groups)))].sum(1)
        summary[col]=dict(flights=len(diff),date_clusters=len(groups),mean=diff[col].mean(),interval95=np.quantile(boots[:,0]/boots[:,1],[.025,.975]).tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','holdout'],required=True);run(ap.parse_args().phase)
