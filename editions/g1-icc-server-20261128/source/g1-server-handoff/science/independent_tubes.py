"""Fit RAFT angular widths, calibrate RAFT and IMM by independent physical world."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import pickle
import numpy as np
import torch
from lightgbm import LGBMRegressor
from server_paths import RUN
from worlds import SPLITS,load_world
from grudata import schedule
from grutrain import GRID
from age_panel import COUNTS
from common import truth
from raft_rebuild import Raft
from routeclock.filters import PositionIMM
from routeclock.geometry import angles,angle_error,angular_jacobian

H=np.array([.5,1,2,3,5,8,10])
DENSE=np.arange(501)*.02


@lru_cache(maxsize=1)
def raft():
    torch.set_num_threads(1)
    return Raft()


def raft_features(hist,c):
    o=hist[-1];diff=np.abs(angle_error(angles(c),angles(o.position)))
    return np.column_stack([np.repeat(GRID,2),np.tile([0.,1.],21),diff.ravel(),
         np.full(42,np.log1p(len(hist))),np.full(42,hist[-1].t-hist[-2].t if len(hist)>1 else 0.),
         np.full(42,np.linalg.norm(o.position)),np.full(42,np.trace(o.covariance))])


def imm_center_width(imm):
    c,cov=imm.forecast(GRID)
    j=np.array([angular_jacobian(p) for p in c])
    raw=np.sqrt(np.maximum(1e-12,np.einsum('hij,hjk,hik->hi',j,cov,j)))
    return c,raw


def interp(curve):
    return np.stack([np.interp(DENSE,GRID,col) for col in curve.T],axis=-1)


def ratio(w,u,t,center,width):
    err=np.abs(angle_error(angles(truth(w,u,t+DENSE)),angles(interp(center))))
    return np.maximum.accumulate((err/np.maximum(interp(width),1e-6)).max(1))[(H*50).astype(int)]


def one(job):
    split,k=job
    w=load_world(split,k);records=list(schedule(w,41000+k))
    hist=[[] for _ in range(w.n)]
    if split=='tube_fit':
        hs=[];vals=[]
        for u,t in records:
            hist[u].append(w.observe(u,t))
            if len(hist[u]) in COUNTS and t+10<=180:
                hs.append(hist[u].copy());vals.append(truth(w,u,t+GRID))
        x=[];y=[]
        for j in range(0,len(hs),48):
            pred=raft().predict(hs[j:j+48])
            for h,c,v in zip(hs[j:j+48],pred,vals[j:j+48]):
                x.append(raft_features(h,c))
                y.append(np.log(np.maximum(1e-6,abs(angle_error(angles(c),angles(v))))).ravel())
        return np.concatenate(x),np.concatenate(y)
    valid=[i for i,(_,t) in enumerate(records) if t+10<=180]
    idx=int(np.random.default_rng(271127+k).choice(valid))
    u,t=records[idx];imm=PositionIMM()
    for user,s in records[:idx+1]:
        if user==u:
            o=w.observe(u,s);hist[u].append(o);imm.observe(o)
    h=hist[u];rc=raft().predict([h])[0];ic,iw=imm_center_width(imm)
    with (RUN/'models/raft_width.pkl').open('rb') as f:tree=pickle.load(f)
    rw=np.exp(np.clip(tree.predict(raft_features(h,rc),num_threads=1),-25,3)).reshape(21,2)
    return dict(k=k,u=u,t=t,n=len(h),RAFT=ratio(w,u,t,rc,rw).tolist(),IMM=ratio(w,u,t,ic,iw).tolist())


def main():
    with ProcessPoolExecutor(6) as pool:
        fit=list(pool.map(one,[('tube_fit',k) for k in SPLITS['tube_fit']]))
        x,y=[np.concatenate(a) for a in zip(*fit)]
        model=LGBMRegressor(objective='quantile',alpha=.9,n_estimators=110,num_leaves=12,min_child_samples=85,
                learning_rate=.055,random_state=7,verbosity=-1,n_jobs=2).fit(x,y)
        with (RUN/'models/raft_width.pkl').open('wb') as f:pickle.dump(model.booster_,f)
        groups={}
        for split in ('tube_calibration','rebuild_holdout'):
            groups[split]=list(pool.map(one,[(split,k) for k in SPLITS[split]]))
    n=len(groups['tube_calibration'])
    rank=min(n,int(np.ceil(.9*(n+1))))
    res={}
    for name in ('RAFT','IMM'):
        factors=np.sort([r[name] for r in groups['tube_calibration']],axis=0)[rank-1]
        vals=np.array([r[name] for r in groups['rebuild_holdout']])
        res[name]=dict(factors=factors.tolist(),holdout_coverage=(vals<=factors).mean(0).tolist())
    (RUN/'models/independent_tubes.json').write_text(json.dumps(dict(split_worlds={s:list(SPLITS[s]) for s in groups},
       rank=rank,rank_denominator=n,target=.9,unthinned_horizons_s=H.tolist(),dense_step_s=.02,
       fit_rows=len(x),result=res,anchors=groups),indent=2))
    print(res)


if __name__=='__main__':main()
