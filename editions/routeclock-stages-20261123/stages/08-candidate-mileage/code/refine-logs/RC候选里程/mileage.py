"""Candidate-conditioned distance and supervised monotone arc labels."""
import json
import numpy as np
import torch
from torch import nn
from protocol_local import OUT,OLD,log
import learning as old
from refine.samples import records,paths
MODES=['continued','distance','tempo','both']
def tempo(row,h):
    profile=np.zeros(5,np.float32)
    if not row['valid'][3]:return profile
    h=h[-181:];tt=np.array([v.t-h[-1].t for v in h])
    start=int(np.argmin(abs(tt+float(row['meta'][3,1]))))
    if abs(tt[start]+row['meta'][3,1])>.0002:raise ValueError('historical match clock')
    part=h[start:np.searchsorted(tt,tt[start]+12.,side='right')]
    times=tt[start:start+len(part)]-tt[start]
    if len(times)<2:return profile
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff([v.position for v in part],axis=0),axis=1))]
    fixed=np.arange(0,11,2)
    d=np.interp(fixed,times,arc)+np.maximum(fixed-times[-1],0)*row['v0']
    return (np.diff(d)/2).clip(0,12).astype(np.float32)
def offline(split,dev):
    data=old.read(split,dev);vel=[]
    for f in paths(split):
        if split=='train' and int(f.stem)%5==0:continue
        with np.load(OLD/'packed'/split/f.name) as z:
            for i,(h,_) in enumerate(records(f)):
                vel.append(tempo({k:z[k][i] for k in ('valid','meta','v0')},h))
    data['past']=torch.tensor(np.stack(vel),device=dev)
    assert len(data['past'])==len(data['x'])
    return data
@torch.no_grad()
def label(d):
    # Integer bins of 2 m. Start at zero, 0..6 m per half-second, valid for all four paths.
    y=d['truth']-d['truth'][:,:1]
    cost=(d['curves'][:,:,None]-y[:,None,:,None]).norm(dim=-1)
    n,k,h,g=cost.shape
    dp=torch.full_like(cost[:,:,0],1e8);dp[:,:,0]=0;back=[]
    for j in range(1,h):
        choices=torch.stack([nn.functional.pad(dp,(i,0),value=1e8)[:,:,:g] for i in range(4)],-1)
        low,arg=choices.min(-1);dp=low+cost[:,:,j].clamp_max(25).square();back.append(arg)
    ix=dp.argmin(-1,keepdim=True);seq=[ix]
    for prev in reversed(back):
        ix=ix-prev.gather(-1,ix);seq.append(ix)
    grid=torch.cat(seq[::-1],-1)
    residual=cost.gather(-1,grid[...,None]).squeeze(-1)
    # Downweight off-map trajectories and ambiguous matches at a distant arc coordinate.
    delta=(torch.arange(g,device=grid.device)-grid[...,None]).abs()
    other=cost.masked_fill(delta<4,1e5).amin(-1)
    weight=torch.exp(-residual/4)*((other-residual)/3).clamp(.05,1)*d['valid'][:,:,None]
    return 2*grid,weight
class Mileage(old.Decoder):
    def __init__(self,width,variant):
        super().__init__(width,'joint');self.variant=variant
        self.adjust=nn.Sequential(nn.Linear(width+5+12+4+5,width),nn.SiLU(),nn.Linear(width,6))
        nn.init.zeros_(self.adjust[-1].weight);nn.init.zeros_(self.adjust[-1].bias)
    def forward(self,d,detail=False):
        def norm(k):return ((d[k]-getattr(self,k+'_mean'))/getattr(self,k+'_std')).clamp(-10,10)
        _,state=self.gru(norm('seq'));ctx=self.context(torch.cat([state[-1],norm('x')],-1))
        n=len(ctx);curve=d['curves'];typ=torch.eye(4,device=ctx.device)[None].expand(n,-1,-1)
        x=torch.cat([ctx[:,None].expand(-1,4,-1),norm('meta'),curve[:,:,[2,5,10,20]].reshape(n,4,12)/20,typ],-1)
        speed=(d['v0'][:,None]+7*self.travel(ctx).tanh()).clamp(0,12)[:,None,:].expand(-1,4,-1)
        if self.variant in ('tempo','both'):
            past=d['past'][:,None,:].expand(-1,4,-1)*typ[...,3:4]
            add=self.adjust(torch.cat([x,past/12],-1))
            speed=(speed+5*add[...,:5].tanh()+typ[...,3:4]*add[...,5:].tanh()*(past-speed)).clamp(0,12)
        dist=torch.cat([speed[:,:,:1]*0,(speed.repeat_interleave(4,-1)*.5).cumsum(-1)],-1)
        u=(dist/2).clamp(0,59.9999);a=u.long();ix=a[...,None].expand(-1,-1,-1,3)
        pts=curve.gather(2,ix)+(curve.gather(2,ix+1)-curve.gather(2,ix))*(u-a)[...,None]
        valid=d['valid'].bool();w=self.choice(x).squeeze(-1).masked_fill(~valid,-1e4).softmax(1)*valid
        q=(pts*w[:,:,None,None]).sum(1)
        alpha=torch.sigmoid(self.trust(ctx)).repeat_interleave(4,1)
        alpha=torch.cat([alpha[:,:1]*0,alpha],1)*d['gate']*valid.any(1)[:,None]
        pred=d['base']+alpha[:,:,None]*(q-(d['pathbase']-d['pathbase'][:,:1]))
        return (pred,dist,w) if detail else pred
def load_model(dev,mode):
    ck=torch.load(OUT/(mode+'.pt'),map_location=dev,weights_only=True)
    m=Mileage(48,mode).to(dev);m.load_state_dict(ck['state']);return m.eval()
def selected(dev):return load_model(dev,json.loads((OUT/'selection.json').read_text())['selected'])
@torch.inference_mode()
def infer(m,rows,hs):
    d={k:torch.tensor(np.stack([r[k] for r in rows]),device=next(m.parameters()).device) for k in old.KEYS if k!='truth'}
    d['past']=torch.tensor(np.stack([tempo(r,h) for r,h in zip(rows,hs,strict=True)]),device=d['x'].device)
    return m(d).cpu().numpy()
def train():
    t=offline('train','cuda');v=offline('validation','cuda')
    labels=[label(old.batches(t,ix)) for ix in torch.arange(len(t['x']),device='cuda').split(128)]
    t['target']=torch.cat([a for a,b in labels]);t['conf']=torch.cat([b for a,b in labels])
    start=old.load('cuda');trace={};best={}
    @torch.no_grad()
    def val(m):
        pred=torch.cat([m(old.batches(v,ix)) for ix in torch.arange(len(v['x']),device='cuda').split(128)])
        return float(old.objective(pred,v['truth']))
    initial=val(start)
    for mode in MODES:
        best[mode]=initial;trace[mode]=[]
        for lr,lam,seed in [(.00035,.035,7),(.0008,.10,11),(.0015,.20,19)]:
            torch.manual_seed(seed);m=Mileage(48,mode).cuda()
            m.load_state_dict(start.state_dict(),strict=False)
            assert abs(val(m)-initial)<.00001
            if not (OUT/(mode+'.pt')).exists():torch.save({'state':m.state_dict(),'epoch':0},OUT/(mode+'.pt'))
            opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.001)
            stale=0;local=initial;hist=[]
            for epoch in range(65):
                for ix in torch.randperm(len(t['x']),device='cuda').split(128):
                    d=old.batches(t,ix);p,dist,w=m(d,True)
                    loss=old.objective(p,d['truth'])
                    if mode in ('both','distance'):
                        j=[2,6,10,16,20];weight=d['conf'][:,:,j]*(.25+.75*w.detach()[:,:,None])
                        residual=nn.functional.smooth_l1_loss(dist[:,:,j],d['target'][:,:,j].float(),reduction='none',beta=2)
                        loss=loss+lam*(residual*weight).sum()/weight.sum().clamp_min(1)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
                value=val(m);hist.append(value)
                if value<local-.00005:
                    local=value;stale=0
                    if value<best[mode]:
                        best[mode]=value;torch.save({'state':m.state_dict(),'seed':seed,'epoch':epoch+1},OUT/(mode+'.pt'))
                else:stale+=1
                if stale==5:
                    for g in opt.param_groups:g['lr']*=.35
                if stale>=10:break
            trace[mode].append(dict(lr=lr,coefficient=lam if mode in ('distance','both') else 0,seed=seed,validation=hist))
            (OUT/'traces.json').write_text(json.dumps(trace))
            log(f'{mode} 种子 {seed} 已微调 {len(hist)} 轮，验证最优 {local:.5f}；上一版 {initial:.5f}。')
    win=min(['distance','tempo','both'],key=best.get)
    (OUT/'selection.json').write_text(json.dumps(dict(selected=win,scores=best,previous=initial)))
    log(f'验证选中 {win}，分数 {best[win]:.5f}，所有权重冻结。')
if __name__=='__main__':train()
