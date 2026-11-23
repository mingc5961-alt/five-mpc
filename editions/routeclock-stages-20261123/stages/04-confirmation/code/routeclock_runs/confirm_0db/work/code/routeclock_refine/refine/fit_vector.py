"""Adapt the velocity GRU, then fit its native angular quantile head out of sample."""
import argparse
import copy
import json
import numpy as np
import torch
from .models import VectorNet, DEVICE, vector_input
from .data import BASE
from .samples import paths,records

def load(files):
    rows = [(vector_input(h),h[-1].position,y[25::25]) for f in files for h,y in records(f)]
    return tuple(torch.tensor(np.stack(items),device=DEVICE,dtype=torch.float32) for items in zip(*rows))

def angular(x):
    return torch.stack([torch.atan2(x[...,1],x[...,0]),torch.atan2(x[...,2],torch.linalg.vector_norm(x[...,:2],dim=-1))],-1)

def loss(net,data,ix,part):
    v,origin,target = [x[ix] for x in data]
    delta,scale = net(v)
    e = angular(delta+origin[:,None])-angular(target)
    error = torch.atan2(e.sin(),e.cos()).abs()
    if part=='center':
        return torch.linalg.vector_norm(error,dim=-1).mean()
    residual = error.detach()-scale
    return torch.maximum(.9*residual,-.1*residual).mean()

def fit_stage(net,train,val,part,lr,seed):
    optimizer = torch.optim.Adam((p for p in net.parameters() if p.requires_grad),lr=lr)
    generator = np.random.default_rng(seed)
    best, stale, history = float('inf'),0,[]
    state = None
    for epoch in range(50):
        net.train()
        if part=='head':
            net.gru.eval()
        for batch in np.array_split(generator.permutation(len(train[0])),int(np.ceil(len(train[0])/64))):
            optimizer.zero_grad()
            value = loss(net,train,batch,part)
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            scores=[(len(ids),float(loss(net,val,ids,part))) for ids in np.array_split(np.arange(len(val[0])),max(1,int(np.ceil(len(val[0])/64))))]
        score=sum(n*s for n,s in scores)/len(val[0]);history.append(score)
        print(f"{part} lr={lr:g} epoch={epoch+1} validation={score:.7f}",flush=True)
        if score < best:
            best,stale,state = score,0,copy.deepcopy(net.state_dict())
        else:
            stale+=1
            if stale>=5: break
    if state is None: raise ValueError('nonfinite validation loss')
    return best,state,history

def main(seed):
    files=paths('train');valid=paths('validation')
    assert len(files)==1200 and len(valid)==400
    train=load([f for f in files if int(f.stem)%5]);held=load([f for f in files if int(f.stem)%5==0]);val=load(valid)
    results={};winner=float('inf')
    for lr in [.001,.0003]:
        torch.manual_seed(seed)
        net=VectorNet().to(DEVICE)
        net.head.requires_grad_(False)
        best,state,history=fit_stage(net,train,val,'center',lr,seed)
        results[str(lr)]=history
        if best<winner: winner,center=best,state
    net.load_state_dict(center)
    net.gru.requires_grad_(False);net.head.requires_grad_(True)
    _,final,results['head']=fit_stage(net,held,val,'head',.001,seed)
    out=BASE/'weights';out.mkdir(exist_ok=True)
    path=out/f'vector{seed}.pt'
    torch.save(final,path.with_suffix('.part'));path.with_suffix('.part').replace(path)
    (out/f'vector{seed}_fit.json').write_text(json.dumps(dict(seed=seed,center_train_routes=960,head_train_routes=240,validation_routes=400,validation_loss_rad=results)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=7);main(p.parse_args().seed)
