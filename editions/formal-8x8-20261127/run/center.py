import functools,pickle,json
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from base import *
from routeclock.config import load_config
from routeclock.filters import PositionIMM
from routeclock.memory import RouteMemory
from routeclock.predictors import Tracker,apply_gate
from routeclock.geometry import angles,angle_error
from refine.models import HS
from features import common,rc_features,rc_params
from fit_heads import predict
from train_chronos_long import context,CHECKPOINT
from advanced_tubes import Predictor,Chronos2Pipeline,ROOT
import replay_new as replay
from core import pack
from learning import infer,load
CFG=load_config()
@functools.lru_cache(maxsize=2)
def read_pickle(p):return pickle.loads(p.read_bytes())
def window(h,t):return h[np.searchsorted([o.t for o in h],h[-1].t-t-1e-9):]
def replay_imm(h):
    imm=PositionIMM()
    for o in h:imm.observe(o)
    return imm
def route_state(h):
    i=PositionIMM();r=RouteMemory(CFG['models']['route'])
    for o in h:
        i.observe(o);r.update(o,i.state()[0][3:])
    return i.forecast(HS)[0],r.forecast(HS)[0]
def features(h,n):
    o=h[-1];t=np.array([v.t for v in h]);p=np.array([v.position for v in h])
    x,mask,_=context(t,p,t[-1]);k=2*n
    return np.column_stack([x[-k:]/30,mask[-k:],np.linspace(-1,0,k)]).astype('float32')
class Residual(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name
        if name.startswith('G'):self.encoder=nn.GRU(5,48,batch_first=True)
        else:
            self.embed=nn.Linear(5,48)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(48,4,96,dropout=0.,batch_first=True),2)
        self.out=nn.Sequential(nn.Linear(48+6,64),nn.SiLU(),nn.Linear(64,60))
    def forward(self,x,i):
        if self.name.startswith('G'):z=self.encoder(x)[1][-1]
        else:z=self.encoder(self.embed(x),src_key_padding_mask=x[:,:,3]<.5)[:,-1]
        state=torch.cat((i[:,0]/250,(i[:,2]-i[:,0])/5),1)
        y=self.out(torch.cat((z,state),1)).reshape(-1,20,3)
        return i+torch.cat((y[:,:1]*0,y),1)
def netname(arm):return {'T8P':'T8','G8P':'G8'}.get(arm,arm)
def modelpath(arm,version):return BASE/'weights'/f'{arm}_{version}.pt'
@functools.lru_cache(maxsize=4)
def learned(name,version,device="cpu"):
    m=Residual(name);m.load_state_dict(torch.load(modelpath(name,version),weights_only=True,map_location='cpu'));return m.to(device).eval()
def short(name,version,h,i,device="cpu"):
    features_tensor = torch.as_tensor(features(h,int(name[1:])),device=device)[None]
    initial = torch.as_tensor(i.astype('float32'),device=device)[None]
    return learned(name,version,device)(features_tensor,initial)[0].detach().cpu().numpy()
def delayed_evidence(h,name,version,short_history=None,device="cpu"):
    t=np.array([o.t for o in h]);idx=np.searchsorted(t,t[-1]-10.,side='right')-1
    e=np.zeros(21)
    if idx<1:return e
    old=h[:idx+1];i,r=route_state(old)
    sh=old if short_history is None else [o for o in short_history if o.t<=old[-1].t]
    if short_history is not None:i=replay_imm(sh).forecast(HS)[0]
    s=short(name,version,sh,i,device)
    # Score issued forecasts only against subsequently paid observations.
    for k in range(1,21):
        j=np.searchsorted(t,t[idx]+HS[k]);candidates=[v for v in (j-1,j) if idx<v<len(t)]
        if not candidates:continue
        j=min(candidates,key=lambda j:abs(t[j]-t[idx]-HS[k]))
        age=t[j]-t[idx]
        if abs(age-HS[k])>1.5 or age>10:continue
        truth=angles(h[j].position)
        def loss(path):
            pt=np.array([np.interp(age,HS,col) for col in path.T]);return np.linalg.norm(angle_error(angles(pt),truth))
        e[k]=np.clip(np.log((loss(s)+.001)/(loss(r)+.001)),-4,4)
    return e
def tube_file(a,H,v,mode):return BASE/'weights'/f'tube_{a}_{H}_{v}_{mode}.npz'
@functools.lru_cache(maxsize=3)
def advanced(arm):
    if arm=='Czero':
        m=Predictor.__new__(Predictor);m.name='Chronos-2';m.head=None
        m.net=Chronos2Pipeline.from_pretrained(str(CHECKPOINT),device_map='cuda',dtype=torch.float32).model.eval()
    else:m=Predictor('Chronos-2' if arm in ('Cnative','Cexternal') else arm,7)
    return m
@functools.lru_cache(maxsize=1)
def joint_net():return load('cpu')
@functools.lru_cache(maxsize=1)
def joint_head():return replay.head_for('joint-refit')[0]
class System:
    def __init__(self,a,H=180,version='arch',mode='all',inference_device='cpu'):
        self.inference_device=inference_device
        self.a=a;self.H=H;self.version=version;self.mode=mode
        self.native=None;self.aux=None;self.last={}
    def joint(self,h,full,key):
        # Rebuild when the left edge moves; otherwise continue only a common prefix.
        tr=self.last.get(key)
        if tr is None or tr.history[0] is not h[0] or len(tr.history)>len(h) or h[len(tr.history)-1] is not tr.history[-1]:
            tr=Tracker(CFG);self.last[key]=tr
        for o in h[len(tr.history):]:tr.observe(o);tr.issue()
        p=tr.snapshots[-1];apply_gate(p,rc_params_cached(),p.features,'RC','P')
        row=pack(h,tr.memory,tr.imm,p)
        if self.mode=='route' and len(full)>len(h):
            i=replay_imm(full).forecast(HS)[0];w=p.diagnostics['gate_weights']['RC'][:,None]
            row['base']=(w*p.centers['P']+(1-w)*i).astype('float32')
        return infer(joint_net(),[row],'joint')[0],rc_features(p,p.features)
    @torch.inference_mode()
    def raw(self,histories):
        a=self.a;hs=[window(h,self.H) for h in histories]
        if a in ('Czero','Cnative','Cexternal','VECTOR','RAFT'):
            result=advanced(a).batch(hs);cs=np.stack([c for c,s in result]);width=np.stack([s for c,s in result])
            if a in ('VECTOR','RAFT'):
                b=read_pickle(ROOT/'tubes'/a/'7/bundle.pkl')
                width=np.broadcast_to(b['fixed'],width.shape)
            if a=='Cexternal':
                hd=read_pickle(replay.rr.SPECIAL/'heads/Chronos-2/common.pkl')
                width=predict(hd,np.stack([common(h,c) for h,c in zip(hs,cs)]),None)
            return cs,width
        if a=='Joint07':
            rows=[self.joint(h,window(full,180),id(full)) for h,full in zip(hs,histories)]
            # Offline calibration histories are typically new list objects.
            if len(self.last)>32:self.last.clear()
            cs=np.stack([c for c,z in rows]);hd=joint_head()
            return cs,predict(hd,np.stack([common(h,c) for h,c in zip(hs,cs)]),np.stack([z for c,z in rows]))
        cs=[];parts=[]
        for h,full in zip(hs,histories):
            sh=window(full,180) if self.mode=='route' else h
            if a in ('C','P','T8P','G8P'):
                i,r=route_state(h)
                if self.mode=='route':i=replay_imm(sh).forecast(HS)[0]
            else:i=replay_imm(sh).forecast(HS)[0]
            if a in ('C','P'):c=r
            elif a=='IMM':c=i
            else:
                n=netname(a);s=short(n,self.version,sh,i,self.inference_device);c=s
                if a.endswith('P'):
                    e=delayed_evidence(h,n,self.version,sh if self.mode=='route' else None,self.inference_device)
                    prior,beta=gate_params(a,self.version)
                    w=1/(1+np.exp(-np.clip(prior+beta*e,-30,30)))
                    c=s+w[:,None]*(r-s);parts.append((s,r,w))
            cs.append(c)
        self.aux=parts;return np.stack(cs),np.ones((len(cs),21,2))
    def batch(self,histories):
        cs,width=self.raw(histories)
        if self.a in ('C','IMM','T8','T180','G8','G180','T8P','G8P'):
            width=np.broadcast_to(self.shape,width.shape)
        return list(zip(cs,width))
    def setup_tube(self,transfer=False):
        H=180 if transfer else self.H
        mode='all' if transfer else self.mode
        with np.load(tube_file(self.a,H,self.version,mode)) as z:self.shape=z['shape'];self.factors=z['factors']
@functools.lru_cache(maxsize=1)
def rc_params_cached():return rc_params()['gate']
@functools.lru_cache(maxsize=4)
def gate_params(a,v):
    with np.load(BASE/'weights'/f'gate_{a}_{v}.npz') as z:return z['prior'],z['beta']
