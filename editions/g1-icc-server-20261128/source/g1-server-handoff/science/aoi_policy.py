"""Paid-state AoI policy, legal-action masks and target values for the two DQN protocols."""
from dataclasses import dataclass
import hashlib
import json
import sys
import numpy as np
import torch
from torch import nn
from server_paths import ROOT,RUN
from worlds import load_world
sys.path.append(str(ROOT/'tools'))
from aoi_reference import QuotaGrid


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.recipient=nn.Sequential(nn.Linear(64*2+5,64),nn.ReLU(),nn.Linear(64,1))
        self.wait=nn.Sequential(nn.Linear(64+5,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,users,public):
        enc=self.encode(users)
        pooled=torch.cat([enc.mean(dim=1),public],-1)
        local=self.recipient(torch.cat([enc,pooled[:,None].expand(-1,users.shape[1],-1)],-1))
        return torch.cat([local.squeeze(-1),self.wait(pooled)],-1)


def choices(q,legal):
    if not legal.any(dim=-1).all():raise ValueError('nonterminal state has no permitted actions')
    return q.masked_fill(~legal,-torch.inf).argmax(dim=-1)


@torch.no_grad()
def td_target(net,users,public,legal,reward,terminal,gamma=.99):
    result=reward.clone()
    active=~terminal
    if active.any():
        q=net(users[active],public[active])
        pick=choices(q,legal[active])
        result[active]+=gamma*q.gather(1,pick[:,None]).flatten()
    return result


@dataclass
class PaidState:
    grid: QuotaGrid
    protocol: str

    def __post_init__(self):
        if self.protocol not in ('fixed','joint'):raise ValueError(self.protocol)
        self.receipts=[None]*self.grid.users
        self.used=0;self.queue=0.;self.tick=0
        self.fixed=set(self.grid.fixed_indices()) if self.protocol=='fixed' else set()
        room=self.grid.opportunities-self.grid.users
        self.alpha=(self.grid.quota-self.grid.users)/room if room else 0.

    def view(self):
        g=self.grid;i=self.tick
        start=g.start_us(i)/1e6
        x=np.zeros((g.users,10),dtype=np.float32)
        for u,o in enumerate(self.receipts):
            if o is not None:
                x[u]=[*np.asarray(o.position)/300.,o.radial_velocity/12.,*np.sqrt(np.diag(o.covariance)),
                      o.radial_std,(start-o.t)/10.,1.]
        public=np.array([self.queue/max(1,g.quota), (g.quota-self.used)/g.quota,
            (g.opportunities-i)/g.opportunities,g.quota/g.opportunities,g.fix_us/g.budget_us*g.service_us/1e6],
            dtype=np.float32)
        legal=np.zeros(g.users+1,dtype=bool)
        if i<g.users:legal[i]=True
        elif self.protocol=='fixed':
            if i in self.fixed:legal[:g.users]=True
            else:legal[-1]=True
        else:
            wait,sense=g.allowed(i,self.used)
            legal[-1]=wait;legal[:g.users]=sense
        return x,public,legal

    def step(self,action,observe,legal):
        if action<0 or action>=len(legal) or not legal[action]:raise ValueError('illegal action')
        g=self.grid;i=self.tick;purchased=(action<g.users)
        if purchased:
            end=g.start_us(i)+g.fix_us
            self.receipts[action]=observe(action,end/1e6)
            self.used+=1
        else:end=None
        if i>=g.users:self.queue=max(0.,self.queue+int(purchased)-self.alpha)
        self.tick+=1
        return end


@torch.inference_mode()
def rollout(grid,protocol,net,observe):
    state=PaidState(grid,protocol);events=[]
    while state.tick<grid.opportunities:
        u,p,mask=state.view()
        if mask.sum()==1:action=int(np.flatnonzero(mask)[0])
        else:
            q=net(torch.from_numpy(u)[None],torch.from_numpy(p)[None])
            action=int(choices(q,torch.from_numpy(mask)[None])[0])
        end=state.step(action,observe,mask)
        if end is not None:events.append((action,end-grid.fix_us,end))
    assert len(events)==grid.quota and [e[0] for e in events[:grid.users]]==list(range(grid.users))
    return events


def smoke():
    torch.manual_seed(53);torch.set_num_threads(1);net=Policy().eval()
    w=load_world('development',1400);out=[]
    for b,k in [(3,700),(3,950),(4.5,950),(9,950)]:
        g=QuotaGrid(budget_us=round(b*1e6),quota=k)
        for protocol in ('fixed','joint'):
            a=rollout(g,protocol,net,w.observe);other=rollout(g,protocol,net,w.observe)
            assert a==other
            if protocol=='fixed':assert [t for _,t,_ in a]==[g.start_us(i) for i in g.fixed_indices()]
            out.append(dict(b=b,k=k,protocol=protocol,time_s=len(a)*g.fix_us/1e6,
                       digest=hashlib.sha256(json.dumps(a,separators=(',',':')).encode()).hexdigest()))
    x=torch.randn(3,16,10);p=torch.randn(3,5);m=torch.zeros(3,17,dtype=torch.bool)
    m[0,0]=True;m[1,-1]=True
    r=torch.tensor([1.,2.,3.]);done=torch.tensor([False,False,True])
    v=td_target(net,x,p,m,r,done)
    q=net(x,p)
    torch.testing.assert_close(v,torch.stack([1+.99*q[0,0],2+.99*q[1,-1],r[2]]))
    perm=torch.randperm(16);swap=net(x[:,perm],p)
    torch.testing.assert_close(swap[:,:16],q[:,:16][:,perm]);torch.testing.assert_close(swap[:,-1],q[:,-1])
    (RUN/'aoi_interface_audit.json').write_text(json.dumps(dict(world=1400,random_initialization=53,
       test='interface only, no optimization or performance comparison',cases=out,
       terminal_and_bootstrap_mask=True,permutation_equivariant=True),indent=2))
    print('8 protocol/budget cases, deterministic event counts, Q masks and permutation check passed')


if __name__=='__main__':smoke()
