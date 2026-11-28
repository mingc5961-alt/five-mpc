"""Last-paid-snapshot controls and common-prefix evaluation of four independent baselines."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from server_paths import RUN
from worlds import SPLITS,load_world
from common import truth
from age_panel import AGES,COUNTS,angle_deg
from center import mean_projection
from routeclock.filters import PositionIMM
from grudata import schedule
from raft_rebuild import Raft
from paid_full import chrono

OUT=RUN/'independent_panel'
FEATURES=['x_m','y_m','z_m','radial_mps','cov_xx','cov_yy','cov_zz','radial_std','age_s','extra_s']


def features(o,age,h):
    return [*o.position,o.radial_velocity,*np.diag(o.covariance),o.radial_std,age,h]


def unit(x):
    return x/np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-9)


def train_world(job):
    split,k=job
    w=load_world(split,k)
    count=np.zeros(w.n,int)
    x=[];y=[]
    for u,t in schedule(w,41000+k):
        count[u]+=1
        if count[u] not in COUNTS:continue
        o=w.observe(u,t)
        for age in AGES:
            for h in (0,2):
                if age+h>10 or t+age+h>180:continue
                x.append(features(o,age,h))
                y.append(unit(truth(w,u,t+age+h)))
    return np.asarray(x,np.float32),np.asarray(y,np.float32)


class Snapshot(nn.Module):
    def __init__(self,age):
        super().__init__()
        self.age=age
        self.register_buffer('mu',torch.zeros(len(FEATURES)))
        self.register_buffer('sigma',torch.ones(len(FEATURES)))
        self.head=nn.Sequential(nn.Linear(10 if age else 9,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,3))

    def forward(self,x):
        z=(x-self.mu)/self.sigma
        if not self.age:z=torch.cat([z[:,:-2],z[:,-1:]],dim=-1)
        base=nn.functional.normalize(x[:,:3])
        return nn.functional.normalize(base+.05*self.head(z))


def train():
    torch.set_num_threads(2)
    data={}
    with ProcessPoolExecutor(8) as pool:
        for split in ('train','train_validation'):
            rows=list(pool.map(train_world,[(split,k) for k in SPLITS[split]]))
            data[split]=[torch.from_numpy(np.concatenate(col)) for col in zip(*rows)]
    x,y=data['train'];vx,vy=data['train_validation']
    results={}
    for age in (True,False):
        torch.manual_seed(271127);net=Snapshot(age)
        net.mu.copy_(x.mean(0));net.sigma.copy_(x.std(0).clamp_min(.001))
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        best=float('inf');curve=[]
        for epoch in range(1,19):
            for ix in torch.randperm(len(x)).split(1024):
                loss=torch.linalg.vector_norm(net(x[ix])-y[ix],dim=1).mean()
                opt.zero_grad();loss.backward();opt.step()
            with torch.inference_mode():
                v=sum(torch.linalg.vector_norm(net(a)-b,dim=1).sum().item()
                      for a,b in zip(vx.split(2048),vy.split(2048)))/len(vx)
            curve.append(v)
            if v<best:
                best=v;torch.save(net.state_dict(),RUN/'models'/f'snapshot_age_{int(age)}.pt')
        results[str(age)]=curve
    (RUN/'models/snapshot_fit.json').write_text(json.dumps(dict(features=FEATURES,seed=271127,train=len(x),
       validation=len(vx),objective='unit LOS chord distance; minimum validation loss',remove='age only, retain extra horizon',
       validation_curves=results),indent=2))


@lru_cache(maxsize=1)
def models():
    torch.set_num_threads(1)
    snaps=[]
    for age in (True,False):
        net=Snapshot(age)
        net.load_state_dict(torch.load(RUN/'models'/f'snapshot_age_{int(age)}.pt',weights_only=True))
        snaps.append(net.eval())
    return Raft(),chrono(),snaps


def score(k):
    out=OUT/f'{k}.csv.gz'
    if out.exists():return
    w=load_world('rebuild_holdout',k)
    raft,cnet,snaps=models()
    ims=[PositionIMM() for _ in range(w.n)];hist=[[] for _ in ims]
    keys=[];inputs=[];imm=[]
    for u,t in schedule(w,41000+k):
        o=w.observe(u,t);hist[u].append(o);ims[u].observe(o)
        n=len(hist[u])
        if n in COUNTS:
            keys.append((u,n));inputs.append(hist[u].copy())
            imm.append(mean_projection()@(ims[u].weights[:,None]*ims[u].x).ravel())
    curves={}
    for j in range(0,len(inputs),12):
        hs=inputs[j:j+12]
        for i,(rp,(cp,_)) in enumerate(zip(raft.predict(hs),cnet.batch(hs)),j):
            curves[keys[i]]=(hs[i-j][-1],imm[i],rp,cp)
    src=pd.read_csv(RUN/'panel'/f'{k}.csv.gz',usecols=['world','u','n','t','age','h','label','natural'])
    xs=[];ys=[];per=[]
    for q in src.itertuples():
        o,ip,rp,cp=curves[(q.u,q.n)]
        if abs(o.t-q.t)>1e-8:raise ValueError('paid-prefix mismatch')
        ix=round((q.age+q.h)*2);y=truth(w,q.u,q.t+q.age+q.h)
        xs.append(features(o,q.age,q.h));ys.append(y)
        per.append([angle_deg(p[ix],y) for p in (ip,rp,cp)])
    src[['e_IMM','e_RAFT','e_Chronos']]=np.asarray(per)
    with torch.inference_mode():
        x=torch.tensor(xs,dtype=torch.float32)
        for net,name in zip(snaps,('e_with_age','e_no_age')):
            src[name]=angle_deg(net(x).numpy(),np.asarray(ys))
    tmp=out.with_suffix(f'.{os.getpid()}.tmp')
    src.to_csv(tmp,index=False,compression='gzip');tmp.replace(out)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    train()
    with ProcessPoolExecutor(6) as p:
        for i,_ in enumerate(p.map(score,range(4000,4024)),1):print(i,flush=True)
