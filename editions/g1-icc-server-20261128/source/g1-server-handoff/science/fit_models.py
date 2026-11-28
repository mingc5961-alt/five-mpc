"""Fit a new recurrent model and independent tube heads on the registered splits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
import pickle
import numpy as np
import torch
from server_paths import RUN
import paid_full as pf
from common import truth
from grutrain import ResidualGRU, CHECKPOINT, GRID
from grudata import encode, schedule
from worlds import SPLITS, load_world
from routeclock.filters import PositionIMM
from center import mean_projection

def atomic_npz(path,**data):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez(f,**data)
    tmp.replace(path)


def samples(job):
    split,k=job
    path=RUN/'motion'/f'{k}.npz'
    if path.exists():return
    world=load_world(split,k)
    hist=[[] for _ in range(world.n)]
    states=[PositionIMM() for _ in hist]
    rows=[]
    for u,t in schedule(world,41000+k):
        if t+10>180:continue
        o=world.observe(u,t);hist[u].append(o)
        st=states[u];st.observe(o)
        imm=mean_projection()@(st.weights[:,None]*st.x).ravel()
        x,mask=encode(hist[u][-32:])
        rows.append((x,mask,(truth(world,u,t+GRID[1:])-imm[1:]).astype('float32')))
    x,mask,y=map(np.stack,zip(*rows))
    atomic_npz(path,x=x,mask=mask,y=y)


def arrays(split):
    chunks={key:[] for key in ('x','mask','y')}
    for k in SPLITS[split]:
        with np.load(RUN/'motion'/f'{k}.npz') as d:
            for key in chunks:chunks[key].append(d[key])
    return [torch.from_numpy(np.concatenate(v)) for v in chunks.values()]


def fit_gru():
    torch.set_num_threads(1)
    torch.manual_seed(271127)
    model=ResidualGRU()
    x,m,y=arrays('train')
    vx,vm,vy=arrays('train_validation')
    model.mean.copy_(x[m].mean(0));model.std.copy_(x[m].std(0).clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    best=float('inf')
    curve=[]
    CHECKPOINT.parent.mkdir(exist_ok=True,parents=True)
    for epoch in range(1,25):
        model.train()
        for ids in torch.randperm(len(x)).split(384):
            pred=model(x[ids],m[ids])
            loss=torch.nn.functional.smooth_l1_loss(pred,y[ids])
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
        model.eval()
        with torch.inference_mode():
            val=sum(torch.nn.functional.smooth_l1_loss(model(a,b),c,reduction='sum').item()
                    for a,b,c in zip(vx.split(384),vm.split(384),vy.split(384)))/vy.numel()
        curve.append(dict(epoch=epoch,validation_huber=val))
        if val<best:
            best=val
            torch.save(dict(protocol='linux_refit_v1',state=model.state_dict(),epoch=epoch),CHECKPOINT.with_suffix('.tmp'))
            CHECKPOINT.with_suffix('.tmp').replace(CHECKPOINT)
        print('epoch',epoch,round(val,5),flush=True)
    (RUN/'models/gru_fit.json').write_text(json.dumps(dict(train_examples=len(x),validation_examples=len(vx),curve=curve)))


def tracks(job):
    split,k=job
    path=RUN/'tracks'/f'{k}.npz'
    if path.exists():return
    w=load_world(split,k)
    net=pf.model()
    tracks=[pf.State(w.cfg,net) for _ in range(w.n)]
    rows=[]
    for u,t in schedule(w,41000+k):
        o=w.observe(u,t)
        pk=tracks[u].update(o,net)['G']
        short,route,base,support,_,dt,n=pk
        rows.append((u,t,n,short,route,base,support,dt,np.trace(o.covariance),o.position))
    atomic_npz(path,**dict(zip(('u','t','n','s','r','base','support','dt','cov','origin'),map(np.stack,zip(*rows)))))
    return k


def packets(data):
    from types import SimpleNamespace
    for s,r,b,sup,o,dt,n,cov in zip(*(data[key] for key in ('s','r','base','support','origin','dt','n','cov')),strict=True):
        obs=SimpleNamespace(position=o,covariance=np.eye(3)*(cov/3))
        yield s,r,b,sup,obs,dt,n


def fit_static():
    hs=np.array([2,4,10,20]); vals=np.linspace(0,1,21)
    sums=np.zeros((len(vals),4))
    for k in SPLITS['train']:
        w=load_world('train',k)
        with np.load(RUN/'tracks'/f'{k}.npz') as a:
            take=a['t']+10<=180
            s=a['s'][take][:,hs];r=a['r'][take][:,hs]
            y=pf.angle(np.stack([truth(w,u,t+GRID[hs]) for u,t in zip(a['u'][take],a['t'][take])]))
            for j,v in enumerate(vals):
                sums[j]+=np.linalg.norm(pf.difference(pf.angle(s+v*(r-s)),y),axis=-1).mean(0)
    weights=vals[sums.argmin(0)].tolist()
    (RUN/'models/static.json').write_text(json.dumps(dict(weights=weights,training_worlds=list(SPLITS['train']))))
    return weights


def fit_tube(name):
    from lightgbm import LGBMRegressor
    static=json.loads((RUN/'models/static.json').read_text())['weights'] if name=='Fstar' else None
    feats=[];targets=[]
    for k in SPLITS['tube_fit']:
        w=load_world('tube_fit',k)
        with np.load(RUN/'tracks'/f'{k}.npz') as data:
            for pk,u,t in zip(packets(data),data['u'],data['t']):
                if t+10>180:continue
                c,f=pf.parts(pk,name,static)
                y=abs(pf.difference(pf.angle(truth(w,u,t+GRID)),pf.angle(c))).ravel()
                feats.append(f);targets.append(np.log(np.maximum(y,1e-6)))
    model=LGBMRegressor(objective='quantile',alpha=.9,n_estimators=110,num_leaves=12,min_child_samples=85,
                        learning_rate=.055,random_state=7,verbosity=-1,n_jobs=2)
    model.fit(np.concatenate(feats),np.concatenate(targets))
    tree=model.booster_
    scores=[]
    dense=np.arange(501)*.02
    ends=np.rint(pf.network.QUERY_H[1:]*50).astype(int)
    for k in SPLITS['tube_calibration']:
        w=load_world('tube_calibration',k)
        with np.load(RUN/'tracks'/f'{k}.npz') as data:
            # One preselected independent service anchor per calibration world.
            valid=np.flatnonzero(data['t']+10<=180)
            idx=np.random.default_rng(271127+k).choice(valid)
            pk=next(p for i,p in enumerate(packets(data)) if i==idx)
            c,f=pf.parts(pk,name,static)
            wid=pf.raw(tree,f)
            t=data['t'][idx];u=data['u'][idx]
            point=np.stack([np.interp(dense,GRID,col) for col in c.T],axis=-1)
            width=np.stack([np.interp(dense,GRID,col) for col in wid.T],axis=-1)
            err=abs(pf.difference(pf.angle(truth(w,u,t+dense)),pf.angle(point)))
            scores.append(np.maximum.accumulate((err/width).max(1))[ends])
    rank=min(len(scores),int(np.ceil(.9*(len(scores)+1))))-1
    final=np.sort(scores,axis=0)[rank]
    out=RUN/'models'/f'tube_{name}.pkl'
    temp=out.with_suffix('.tmp')
    with temp.open('wb') as f:pickle.dump(dict(protocol='linux_refit_v1',head=name,tree=tree,final=final,
                                   scores=scores,rank=rank+1),f)
    temp.replace(out)
    print('tube',name,final.tolist(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=('motion','gru','tracks','static','tubes'))
    p.add_argument('--splits',nargs='+',choices=list(SPLITS));p.add_argument('--workers',type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.step in ('motion','tracks'):
        jobs=[(s,k) for s in a.splits for k in SPLITS[s]]
        with ProcessPoolExecutor(a.workers) as pool:
            for i,_ in enumerate(pool.map(samples if a.step=='motion' else tracks,jobs),1):
                if i%8==0 or i==len(jobs):print(a.step,i,len(jobs),flush=True)
    elif a.step=='gru':fit_gru()
    elif a.step=='static':fit_static()
    else:
        with ProcessPoolExecutor(3) as pool:list(pool.map(fit_tube,('A','M','Fstar')))
