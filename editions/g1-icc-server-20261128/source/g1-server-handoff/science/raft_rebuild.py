"""CPU training of the Stage-04 multiscale retrieval model on the 7/9 training split."""
import json
import numpy as np
import torch
from server_paths import RUN
from worlds import SPLITS,load_world
from common import truth
from train_chronos_long import context
from train_raft_long import Bank,Model,HORIZONS
from train_vector_long import angles
from grutrain import GRID

BANK_N=(2,8,24,40)
QUERY_N=(1,2,3,4,6,8,12,24,40)


def dataset(split):
    rows=[];ids=[];nn=[]
    for k in SPLITS[split]:
        w=load_world(split,k)
        from grudata import schedule
        events=list(schedule(w,41000+k))
        us=np.array([u for u,t in events]);ts=np.array([t for u,t in events])
        pos=np.stack([w.observe(u,t).position for u,t in events])
        for u in range(16):
            select=us==u
            t=ts[select];pts=pos[select]
            for n in QUERY_N:
                if n>len(t) or t[n-1]+10>180:continue
                x,m,o=context(t[:n],pts[:n],t[n-1])
                rows.append((x,m,o,truth(w,u,t[n-1]+GRID[1:])))
                ids.append(16*k+u);nn.append(n)
    return [torch.tensor(np.stack(s),dtype=torch.float32) for s in zip(*rows)],torch.tensor(ids),np.array(nn)


def neighbors(bank,data,exclude):
    batches=[bank.retrieve(data[0][j:j+96],data[1][j:j+96],exclude[j:j+96]) for j in range(0,len(exclude),96)]
    return [torch.cat(a) for a in zip(*batches)]


def train():
    torch.manual_seed(7);torch.set_num_threads(3)
    train,ids,ns=dataset('train'); val,vid,_=dataset('train_validation')
    sel=np.isin(ns,BANK_N)
    bdata=[d[sel] for d in train];bids=ids[sel]
    bank=Bank(bdata,bids)
    dest=RUN/'models';dest.mkdir(exist_ok=True)
    torch.save(dict(data=bdata,ids=bids,seed=7),dest/'raft_bank.pt')
    near=neighbors(bank,train,ids);vn=neighbors(bank,val,torch.full_like(vid,-1))
    net=Model();optimizer=torch.optim.Adam(net.parameters(),lr=.0003)
    best=float('inf');curve=[]
    def metric(d,ne,ix):
        xyz=net(d[0][ix],[m[ix] for m in ne])+d[2][ix,None]
        delta=angles(xyz[:,HORIZONS])-angles(d[3][ix][:,HORIZONS])
        return torch.linalg.vector_norm(torch.atan2(delta.sin(),delta.cos()),dim=-1).mean()
    for epoch in range(12):
        for ix in torch.randperm(len(ids)).split(64):
            v=metric(train,near,ix)
            optimizer.zero_grad();v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
        with torch.inference_mode():
            v=sum(metric(val,vn,i).item()*len(i) for i in torch.arange(len(vid)).split(256))/len(vid)
        curve.append(v)
        if v<best:
            best=v;torch.save(net.state_dict(),dest/'raft_refit.pt.tmp')
            (dest/'raft_refit.pt.tmp').replace(dest/'raft_refit.pt')
    (dest/'raft_fit.json').write_text(json.dumps(dict(train=len(ids),bank=len(bids),routes=len(bids.unique()),
      validation=len(vid),bank_prefixes=BANK_N,training_worlds=list(SPLITS['train']),curve_rad=curve,seed=7)))


class Raft:
    def __init__(self):
        d=torch.load(RUN/'models/raft_bank.pt',weights_only=True,map_location='cpu')
        self.bank=Bank(d['data'],d['ids'])
        self.net=Model()
        self.net.load_state_dict(torch.load(RUN/'models/raft_refit.pt',weights_only=True,map_location='cpu'))
        self.net.eval()

    @torch.inference_mode()
    def predict(self,histories):
        inp=[context(np.array([a.t for a in h]),np.stack([a.position for a in h]),h[-1].t) for h in histories]
        x,m,o=[torch.tensor(np.stack(col),dtype=torch.float32) for col in zip(*inp)]
        n=self.bank.retrieve(x,m,torch.full((len(x),),-1))
        y=self.net(x,n).numpy()+o.numpy()[:,None]
        return [np.vstack([h[-1].position,p]) if len(h)>1 else np.tile(h[0].position,(21,1))
                for h,p in zip(histories,y)]

if __name__=='__main__':train()
