"""Frozen four-way split; generate only causal model inputs and offline future labels."""
from pathlib import Path
import os,pickle,sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from paid_full import load_config, World,GROUPS,mixture,GRID
from common import truth
from routeclock.filters import PositionIMM
from center import mean_projection
D=Path(__file__).resolve().parent/"g1mix7"
D.mkdir(exist_ok=True)
SPLITS={"train":12,"val":4,"cal":3,"test":8}
def world(split,k):
    path=D/f"{split}{k}.world"
    with path.open("rb") as f:return pickle.load(f)
def schedule(w,k):
    rng=np.random.default_rng(86921+k)
    t=.0031; rnd=0
    while t<180:
        order=np.arange(16) if rnd==0 else rng.permutation(16)
        for u in order:
            if t>180:return
            yield int(u),t
            t+=.186+rng.uniform(0,.075)
        rnd+=1
def encode(history):
    t=history[-1].t
    x=np.zeros((8,7),np.float32); mask=np.zeros(8,bool); prev=None
    for o in history:
        age=t-o.t
        if age<8-1e-8:
            idx=7-int(np.floor(age+1e-8))
            valid=prev is not None and t-prev.t<=8+1e-8
            dx=o.position-prev.position if valid else np.zeros(3)
            x[idx]=[*dx,o.t-prev.t if valid else 0.,o.radial_velocity,np.log1p(np.trace(o.covariance)),o.radial_std]
            mask[idx]=True
        prev=o
    return x,mask
def prepare(job):
    split,k=job
    wp=D/f"{split}{k}.world"; fp=D/f"{split}{k}.npz"
    if not wp.exists():
        cfg=load_config();cfg["simulation"].update(warmup_s=0,service_s=180,sensing_budget=.05/3)
        w=World(cfg,"gru_paid_mix7_v3/"+split,k,groups=[GROUPS[i] for i in np.repeat(np.arange(6),[3,1,1,1,3,7])[np.random.default_rng(9124+k).permutation(16)]])
        with wp.with_suffix(".writing").open("wb") as f:pickle.dump(w,f,protocol=5)
        wp.with_suffix(".writing").replace(wp)
    if split not in ("train","val") or fp.exists():return
    w=world(split,k)
    hs=[[] for _ in range(16)]; fs=[PositionIMM() for _ in hs]
    rows=[]
    for u,t in schedule(w,k):
        o=w.observe(u,t);hs[u].append(o);fs[u].observe(o)
        x,mask=encode(hs[u][-32:])
        i=mean_projection()@(fs[u].weights[:,None]*fs[u].x).reshape(18)
        rows.append((x,mask,i[1:],truth(w,u,t+GRID[1:]),len(hs[u]),u,t))
    np.savez(fp,**{n:a for n,a in zip(["x","mask","imm","y","n","u","t"],map(np.stack,zip(*rows)))})
    print(split,k,len(rows),flush=True)
if __name__=="__main__":
    with ProcessPoolExecutor(3) as p:list(p.map(prepare,[(s,k) for s,n in SPLITS.items() for k in range(n)]))
