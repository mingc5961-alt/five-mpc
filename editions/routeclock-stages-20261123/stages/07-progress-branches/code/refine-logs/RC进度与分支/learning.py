import copy,json,pickle,time,sys
import numpy as np
import torch
from torch import nn
from core import ROOT,log
from refine.data import ARCHIVE
torch.set_num_threads(1)
MODES=('geometry','progress','branch','joint')
KEYS=('seq','x','curves','meta','valid','base','pathbase','gate','v0','truth')
def read(split,device):
    block={k:[] for k in KEYS}
    for f in sorted((ROOT/'packed'/split).glob('*.npz')):
        if split=='train' and int(f.stem)%5==0:continue
        with np.load(f) as row:
            for k in KEYS:
                if k=='pathbase' and k not in row:
                    archived=pickle.loads((ARCHIVE/split/(f.stem+'.pkl')).read_bytes())
                    a=np.stack([r['package'].centers['P'] for r in archived])
                else:a=row[k]
                block[k].append(a)
    return {k:torch.tensor(np.concatenate(a),device=device,dtype=torch.float32) for k,a in block.items()}
def batches(d,ix):return {k:v[ix] for k,v in d.items()}
class Decoder(nn.Module):
    def __init__(self,width,mode):
        super().__init__();self.mode=mode
        self.gru=nn.GRU(5,width,batch_first=True)
        self.context=nn.Sequential(nn.Linear(width+18,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.travel=nn.Linear(width,5)
        self.choice=nn.Sequential(nn.Linear(width+5+12+4,width),nn.SiLU(),nn.Linear(width,1))
        self.trust=nn.Linear(width,5)
        for key,n in [('seq',5),('x',18),('meta',5)]:
            self.register_buffer(key+'_mean',torch.zeros(n));self.register_buffer(key+'_std',torch.ones(n))
        nn.init.zeros_(self.travel.weight);nn.init.zeros_(self.travel.bias)
        nn.init.constant_(self.trust.bias,-1.)
    def normalize(self,d):
        for key in ('seq','x','meta'):
            z=d[key].reshape(-1,d[key].shape[-1])
            if key=='meta':z=z[d['valid'].reshape(-1).bool()]
            getattr(self,key+'_mean').copy_(z.mean(0));getattr(self,key+'_std').copy_(z.std(0).clamp_min(.05))
    def forward(self,d,details=False):
        def normalized(k):return ((d[k]-getattr(self,k+'_mean'))/getattr(self,k+'_std')).clamp(-10,10)
        _,state=self.gru(normalized('seq'));ctx=self.context(torch.cat([state[-1],normalized('x')],-1))
        n=len(ctx);curves=d['curves']
        move=d['v0'][:,None].expand(-1,5)
        if self.mode in ('progress','joint'):move=(move+7*torch.tanh(self.travel(ctx))).clamp(0,12)
        dist=torch.cat([move[:,:1]*0,(move.repeat_interleave(4,1)*.5).cumsum(1)],1)
        u=(dist/2).clamp(0,59.9999);a=u.long();ix=a[:,None,:,None].expand(-1,4,-1,3)
        pts=curves.gather(2,ix)+(curves.gather(2,ix+1)-curves.gather(2,ix))*(u-a)[:,None,:,None]
        valid=d['valid'].bool().clone()
        if self.mode in ('progress','geometry'):valid[:,1:]=False
        typ=torch.eye(4,device=ctx.device)[None].expand(n,-1,-1)
        shape=curves[:,:,[2,5,10,20]].reshape(n,4,12)/20
        scores=self.choice(torch.cat([ctx[:,None].expand(-1,4,-1),normalized('meta'),shape,typ],-1)).squeeze(-1)
        w=torch.softmax(scores.masked_fill(~valid,-1e4),1)*valid
        q=(pts*w[:,:,None,None]).sum(1)
        alpha=torch.sigmoid(self.trust(ctx)).repeat_interleave(4,1)
        alpha=torch.cat([alpha[:,:1]*0,alpha],1)*d['gate']*valid.any(1)[:,None]
        result=d['base']+alpha[:,:,None]*(q-(d['pathbase']-d['pathbase'][:,:1]))
        return (result,w,dist,alpha) if details else result
def angular(p):
    return torch.stack([torch.atan2(p[...,1],p[...,0]),torch.atan2(p[...,2],p[...,:2].norm(dim=-1))],-1)
def objective(p,y):
    diff=angular(p[:,1:])-angular(y[:,1:])
    e=torch.atan2(torch.sin(diff),torch.cos(diff)).norm(dim=-1)
    return e.mean()*40+.04*(p[:,1:]-y[:,1:]).norm(dim=-1).mean()
def load(device,mode='joint'):
    ck=torch.load(ROOT/(mode+'.pt'),map_location=device,weights_only=True)
    mod=Decoder(ck['width'],mode).to(device);mod.load_state_dict(ck['state']);mod.eval()
    return mod
@torch.inference_mode()
def infer(model,rows,mode=None):
    dev=next(model.parameters()).device
    d={k:torch.tensor(np.stack([r[k] for r in rows]),device=dev) for k in KEYS if k!='truth'}
    return model(d).cpu().numpy()
def train():
    dev='cuda'
    t=read('train',dev);v=read('validation',dev)
    assert len(t['x'])==2880 and len(v['x'])==1200
    valid_ix=torch.arange(len(v['x']),device=dev).split(256)
    book={}
    for mode in MODES:
        best=float('inf');book[mode]=[]
        for width,lr,seed in [(32,.002,7),(64,.0015,11),(48,.0008,19)]:
            torch.manual_seed(seed);model=Decoder(width,mode).to(dev);model.normalize(t)
            opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001);trace=[];stale=0;local=float('inf')
            for epoch in range(65):
                for ix in torch.randperm(len(t['x']),device=dev).split(128):
                    d=batches(t,ix);loss=objective(model(d),d['truth']);opt.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
                with torch.no_grad():
                    val=sum(float(objective(model(batches(v,ix)),v['truth'][ix]))*len(ix) for ix in valid_ix)/len(v['x'])
                trace.append(val)
                if val<local-.00005:
                    local=val;stale=0
                    if val<best:
                        best=val;tmp=ROOT/(mode+'.tmp');torch.save(dict(width=width,state=model.state_dict(),seed=seed,epoch=epoch+1,score=val),tmp);tmp.replace(ROOT/(mode+'.pt'))
                else:stale+=1
                if stale==5:
                    for group in opt.param_groups:group['lr']*=.35
                if stale>=10:break
            book[mode].append(dict(width=width,lr=lr,seed=seed,trace=trace))
            (ROOT/'training.json').write_text(json.dumps(book))
            log(f'{mode} 宽度 {width}、种子 {seed}、学习率 {lr}，训练 {len(trace)} 轮，最低验证损失 {min(trace):.5f}；组内冻结值 {best:.5f}。')
if __name__=='__main__':train()
