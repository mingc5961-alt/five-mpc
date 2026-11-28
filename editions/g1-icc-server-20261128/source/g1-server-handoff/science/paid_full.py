"""Paid-position protocol on the installed RouteClock engine.

The G centre and fitted tube belong to linux_refit_v1. Chronos uses the
archived seed-7 weights. All track state belongs to one user in one world.
"""
from functools import lru_cache
import json
import pickle
import numpy as np
import torch
from server_paths import ROOT, BASE, RUN
from common import truth
from center import mean_projection
from routeclock.config import load_config
from routeclock.world import World, GROUPS
from routeclock.geometry import angles as angle, angle_error as difference
from routeclock.radio import Radio
from routeclock.scheduler import TokenBucket, choose_actions
from routeclock.filters import PositionIMM
from routeclock.memory import RouteMemory
from refine import network
from runtime.fast_ledger import DeferredLedger
from parallel_frontier import frontier as efficient_frontier
from refine.models import forecast as chronos_forecast
from native_map import forecast as route_forecast
from grutrain import Cold, GRID

H = np.array([1.,2.,5.,10.])
PROFILE = np.interp(GRID,[0,1,2,5,10],[0,0,0,.05,.10])


def mixture(reusable=9):
    if reusable!=9:
        raise ValueError('the main cold-start mixture has 9 structured users')
    return np.repeat(GROUPS, [3,1,1,1,3,7]).tolist()


def model():
    return Cold('G1')


@lru_cache()
def bundle(name='A'):
    with (RUN/'models'/f'tube_{name}.pkl').open('rb') as f:
        result=pickle.load(f)
    if result['protocol']!='linux_refit_v1' or result['head']!=name:
        raise ValueError('tube provenance mismatch')
    return result


def raw(tree, features):
    flat = features.reshape(-1,features.shape[-1])
    y = np.exp(np.clip(tree.predict(flat,num_threads=1),-25,3))
    return np.maximum(y,1e-6).reshape(*features.shape[:-2],21,2)


class State:
    def __init__(self, cfg, net, arm='G'):
        if arm!='G':
            raise ValueError(arm)
        self.imm=PositionIMM()
        self.memory=RouteMemory(cfg['models']['route'])
        self.history=[]
        self.issued=[]
        self.count=0
        self.num=np.zeros(4)
        self.den=np.zeros(4)
        self.support=np.zeros(4)
        self.net=net

    def update(self, obs, net):
        if self.history and obs.t<=self.history[-1].t:
            raise ValueError('paid timestamps must increase for this user')
        dt=obs.t-self.history[-1].t if self.history else 0.
        decay=2**(-dt/(15+1.5*H))
        self.num*=decay
        self.den*=decay
        self.support*=decay
        self.issued=[(t,s,r) for t,s,r in self.issued if obs.t-t<=10.]
        for t,s,r in self.issued:
            age=obs.t-t
            def sample(c):
                return angle([np.interp(age,GRID,col) for col in c.T])
            short_angle=sample(s)
            # Degrees fix the scale of the bounded projection and its ridge.
            e=np.rad2deg(difference(short_angle,angle(obs.position)))
            d=np.rad2deg(difference(sample(r),short_angle))
            proximity=np.exp(-.5*((age-H)/np.maximum(.5,.35*H))**2)
            mass=proximity/max(1.,np.linalg.norm(e)/8.,np.linalg.norm(d)/8.)
            self.num+=mass*(-e@d)
            self.den+=mass*(d@d)
            self.support+=mass
        self.history.append(obs)
        self.count+=1
        self.imm.observe(obs)
        self.memory.update(obs,self.imm.state()[0][3:])
        short=mean_projection()@(self.imm.weights[:,None]*self.imm.x).ravel()
        short[1:]+=net.residual(self.history)
        route,_,coverage,_,_=route_forecast(self.memory,GRID)
        prior=net.prior[[2,4,10,20]]
        fitted=np.clip((self.num+.3*prior)/(self.den+.3),0,1)
        w=prior+self.support/(self.support+3)*(fitted-prior)
        weights=np.interp(GRID,H,w)
        self.issued.append((obs.t,short,route))
        return {'G':(short,route,weights,coverage,obs,dt,self.count)}


def parts(pk, variant='A', static=None):
    short,route,base,support,obs,dt,n=pk
    w=np.clip(base+PROFILE,0,1)
    if variant=='F0':
        w=np.interp(GRID,H,[.25,.25,.35,.50])+PROFILE
    elif variant=='Fstar':
        w=np.interp(GRID,H,static)
    elif variant in ('M','S'):
        w=np.zeros_like(w)
    elif variant!='A':
        raise ValueError(variant)
    center=short+w[:,None]*(route-short)
    diff=np.abs(difference(angle(route),angle(short)))
    shift=np.abs(difference(angle(center),angle(obs.position)))
    if variant in ('M','S'):
        diff=np.zeros_like(diff)
        support=np.zeros_like(w)
    repeat=lambda v:np.repeat(v,2)
    cols=[repeat(GRID),np.tile([0.,1.],21),diff.ravel(),shift.ravel(),repeat(support),
          repeat(w),np.full(42,dt),np.full(42,np.log1p(n)),np.full(42,np.trace(obs.covariance))]
    return center,np.column_stack(cols)


def forecast(pk, arm='G', profile=1):
    if arm!='G' or profile!=1:
        raise ValueError((arm,profile))
    return parts(pk)


class ChronosCPU:
    def __init__(self):
        from chronos import Chronos2Pipeline
        self.torch=torch
        self.net=Chronos2Pipeline.from_pretrained('/data/chen/routeclock_inputs_20261120/chronos',device_map='cpu',dtype=torch.float32).model
        self.net.load_state_dict(torch.load(ROOT/'reference/models/Chronos/best.pt',map_location='cpu',weights_only=True))
        self.net.eval()
        self.quantiles=[list(self.net.chronos_config.quantiles).index(q) for q in (.1,.5,.9)]

    @torch.inference_mode()
    def batch(self,histories):
        inputs=[]; masks=[]
        for h in histories:
            times=np.array([v.t for v in h])
            xyz=np.stack([v.position for v in h])
            grid=times[-1]-.5*np.arange(359,-1,-1)
            idx=times.searchsorted(grid,side='right')-1
            mask=(idx>=0)&(times[idx.clip(0)]>grid-.5)
            inputs.append(np.where(mask[:,None],xyz[idx.clip(0)]-xyz[-1],0))
            masks.append(mask)
        x=torch.tensor(np.stack(inputs),dtype=torch.float32)
        mask=torch.tensor(np.stack(masks),dtype=torch.float32)
        n=len(histories)
        out=self.net(context=x.transpose(1,2).reshape(-1,360),context_mask=mask.repeat_interleave(3,0),
                     group_ids=torch.arange(n).repeat_interleave(3),num_output_patches=2)
        q=out.quantile_preds[:,self.quantiles,:20].reshape(n,3,3,20).permute(0,3,1,2).numpy()
        return [chronos_forecast(a+h[-1].position[None,:,None],h[-1]) for h,a in zip(histories,q,strict=True)]


@lru_cache(maxsize=1)
def chrono():
    return ChronosCPU()
