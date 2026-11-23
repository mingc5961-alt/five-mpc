"""Route-isolated multiscale RAFT adaptation with shared XYZ neighbor identities."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from train_chronos_long import split_data
from train_vector_long import ROOT, SEEDS, HORIZONS, angles, write

SCALES = (4, 2, 1)

def pool(context, mask, scale):
    weights = mask.reshape(len(mask), -1, scale).float()
    xyz = context.reshape(len(context), -1, scale, 3)
    count = weights.sum(-1)
    return (xyz * weights[..., None]).sum(2) / count.clamp_min(1)[..., None], (count > 0).float()

class Bank:
    def __init__(self, data, ids):
        self.ids = ids
        x, mask, origin, future = data
        self.scales = [pool(x,mask,g) for g in SCALES]
        self.deltas = future - origin[:,None]

    @torch.no_grad()
    def retrieve(self, x, mask, exclude):
        # Correlate jointly over aligned time and XYZ on their common support.
        predictions = []
        for scale, (bank, bm) in zip(SCALES,self.scales):
            q,qm = pool(x,mask,scale)
            dot = q.flatten(1) @ bank.flatten(1).T
            count = qm @ bm.T
            qsum = torch.stack([q[:,:,k] @ bm.T for k in range(3)],-1)
            bsum = torch.stack([qm @ bank[:,:,k].T for k in range(3)],-1)
            denom = count.clamp_min(1)
            dot -= (qsum*bsum).sum(-1)/denom
            qvar = q.square().sum(-1) @ bm.T - qsum.square().sum(-1)/denom
            bvar = qm @ bank.square().sum(-1).T - bsum.square().sum(-1)/denom
            valid = (count>=3) & (qvar>1e-6) & (bvar>1e-6)
            valid &= exclude[:,None] != self.ids[None,:]
            sim = dot / (qvar.clamp_min(1e-6)*bvar.clamp_min(1e-6)).sqrt()
            scores, idx = sim.masked_fill(~valid,-torch.inf).topk(min(20,len(self.ids)),dim=1)
            any_valid = scores.isfinite().any(1)
            assert not ((self.ids[idx]==exclude[:,None]) & scores.isfinite()).any()
            w = torch.softmax(torch.where(any_valid[:,None],scores,0.)/.1,1)*any_valid[:,None]
            delta = (self.deltas[idx] * w[:,:,None,None]).sum(1)
            predictions.append(delta.reshape(len(x),20//scale,scale,3).mean(2))
        return predictions

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_x = nn.Linear(360,20)
        self.retrieval_pred = nn.ModuleList(nn.Linear(20//g,20) for g in SCALES)
        self.linear_pred = nn.Linear(40,20)

    def forward(self,x,neighbors):
        own = self.linear_x(x.transpose(1,2))
        external = sum(layer(y.transpose(1,2)) for layer,y in zip(self.retrieval_pred,neighbors,strict=True))
        return self.linear_pred(torch.cat((own,external),dim=-1)).transpose(1,2)

def route_ids():
    from train_vector_long import OBS
    ids=[]
    for p in sorted((OBS/'train').glob('*.npz')):
        if int(p.stem)%5:
            with np.load(p) as d:
                ids.extend([int(p.stem)]*len(d['origins']))
    return torch.tensor(ids,device="cuda")

def loss(model,bank,data,indices,exclude):
    x,mask,origin,target = (v[indices] for v in data)
    delta = model(x,bank.retrieve(x,mask,exclude))
    error = angles((origin[:,None]+delta)[:,HORIZONS])-angles(target[:,HORIZONS])
    return torch.linalg.vector_norm(torch.atan2(error.sin(),error.cos()),dim=-1).mean()

def main(pilot):
    data, val = split_data("train"),split_data("validation")
    bank = Bank(data,route_ids())
    assert len(bank.ids)==len(data[0])
    for seed in ((7,) if pilot else SEEDS):
        torch.manual_seed(seed)
        folder=ROOT/"training/RAFT"/str(seed)
        folder.mkdir(parents=True,exist_ok=True)
        if not pilot and (folder/"complete.json").is_file():
            continue
        model=Model().cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
        start=time.monotonic()
        best,stale=float("inf"),0
        for epoch in range(1,13):
            model.train()
            batches=torch.randperm(len(data[0]),device="cuda").split(24)
            for step,idx in enumerate(batches,1):
                optimizer.zero_grad(set_to_none=True)
                value=loss(model,bank,data,idx,bank.ids[idx])
                value.backward()
                grad=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if pilot and step==10:
                    torch.cuda.synchronize()
                    print(json.dumps(dict(batches=step,routes=len(bank.ids.unique()),windows=len(bank.ids),loss=float(value.detach()),grad_norm=float(grad),peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,elapsed_s=time.monotonic()-start)))
                    return
                if not pilot and (step==1 or step%10==0):
                    write(folder/"progress.txt",f"第 {epoch}/12 轮，第 {step}/{len(batches)} 批，已用 {time.monotonic()-start:.0f} 秒")
            model.eval()
            with torch.no_grad():
                valid=torch.arange(len(val[0]),device="cuda")
                metric=sum(float(loss(model,bank,val,idx,torch.full_like(idx,-1)))*len(idx) for idx in valid.split(24))/len(valid)
            if not np.isfinite(metric): raise ValueError("nonfinite validation loss")
            with (folder/"curve.jsonl").open("a") as f:
                print(json.dumps(dict(epoch=epoch,validation_rad=metric,elapsed_s=time.monotonic()-start)),file=f)
            if metric<best:
                temp=folder/"best.partial"
                torch.save(model.state_dict(),temp)
                temp.replace(folder/"best.pt")
                best,stale=metric,0
            else: stale+=1
            if stale>=5 or time.monotonic()-start>=3600: break
        with (folder/"best.pt").open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
        write(folder/"complete.json",json.dumps(dict(seed=seed,artifact="center",epochs=epoch,sha256=digest,validation_rad=best)))
        write(folder/"progress.txt",f"中心训练完成，共 {epoch} 轮，最佳验证误差 {best:.5f} rad")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--pilot",action="store_true")
    main(parser.parse_args().pilot)
