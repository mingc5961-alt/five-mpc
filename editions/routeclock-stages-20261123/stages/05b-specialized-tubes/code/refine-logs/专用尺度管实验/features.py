from unittest.mock import patch
import pickle
import numpy as np
import torch
from protocol import ROOT, PRIOR, SEED
import advanced_tubes as adv
from routeclock.geometry import angles,angle_error
from routeclock.predictors import apply_gate
from routeclock.validity import QUERY_H
from refine.models import HS
from train_raft_long import Bank,pool,SCALES

K=np.searchsorted(HS,QUERY_H)
EXPAND=np.searchsorted(QUERY_H,HS)

def repeat(x):
    return np.broadcast_to(x,(len(K),len(x)))

def observation_velocity(h):
    if len(h)<2:
        return np.zeros(3)
    return (h[-1].position-h[-2].position)/(h[-1].t-h[-2].t)

def common(h,c):
    o=h[-1]
    t=np.array([v.t for v in h if v.t>=o.t-180.])
    gaps=np.diff(t)
    last=gaps[-8:] if gaps.size else np.array([0.])
    visible_start=max(0.,o.t-180.)
    longest=max(t[0]-visible_start,last.max(),gaps.max(initial=0.))
    obs=np.r_[o.position,np.diag(o.covariance),o.radial_velocity,o.radial_std,
              len(t),last[-1],last.mean(),last.max(),longest,np.linalg.norm(observation_velocity(h))]
    drift=angle_error(angles(c),angles(c[0]))
    return np.column_stack([QUERY_H,repeat(obs),c[K]-c[0],np.linalg.norm(c[K],axis=1),
                            drift[K],np.maximum.accumulate(abs(drift),axis=0)[K]])

def rc_features(p,base):
    difference=angle_error(angles(p.centers['P']),angles(p.centers['I']))
    return np.column_stack([base[K],difference[K]])

def rc_params():
    return pickle.loads((PRIOR/'cache/classical.pkl').read_bytes())['RC']

class DiagnosticBank(Bank):
    """Use the archived retrieval algebra and expose the selected neighbors before averaging."""
    @torch.no_grad()
    def retrieve(self,x,mask,exclude):
        predictions=[]
        self.diagnostics=[]
        for scale,(bank,bm) in zip(SCALES,self.scales):
            q,qm=pool(x,mask,scale)
            dot=q.flatten(1)@bank.flatten(1).T
            count=qm@bm.T
            qsum=torch.stack([q[:,:,k]@bm.T for k in range(3)],-1)
            bsum=torch.stack([qm@bank[:,:,k].T for k in range(3)],-1)
            denom=count.clamp_min(1)
            dot-=(qsum*bsum).sum(-1)/denom
            qvar=q.square().sum(-1)@bm.T-qsum.square().sum(-1)/denom
            bvar=qm@bank.square().sum(-1).T-bsum.square().sum(-1)/denom
            valid=(count>=3)&(qvar>1e-6)&(bvar>1e-6)&(exclude[:,None]!=self.ids[None,:])
            sim=dot/(qvar.clamp_min(1e-6)*bvar.clamp_min(1e-6)).sqrt()
            scores,idx=sim.masked_fill(~valid,-torch.inf).topk(min(20,len(self.ids)),dim=1)
            any_valid=scores.isfinite().any(1)
            w=torch.softmax(torch.where(any_valid[:,None],scores,0.)/.1,1)*any_valid[:,None]
            deltas=self.deltas[idx]
            delta=(deltas*w[:,:,None,None]).sum(1)
            predictions.append(delta.reshape(len(x),20//scale,scale,3).mean(2))
            finite=scores.isfinite()
            clean=scores.masked_fill(~finite,0.)
            metadata=torch.stack([valid.sum(1),finite.sum(1),(w*count.gather(1,idx)).sum(1),
                                  clean[:,0],torch.where(finite[:,1],clean[:,0]-clean[:,1],0.),
                                  w.max(1).values,torch.where(any_valid,1/w.square().sum(1).clamp_min(1e-12),0.)],1)
            self.diagnostics.append(tuple(v.cpu().numpy() for v in (metadata,w,deltas)))
        return predictions

def raft_features(origin,diag,i):
    features=[];means=[]
    for metadata,weights,deltas in diag:
        w=weights[i]
        future=deltas[i][:,np.maximum(K-1,0)].copy()
        future[:,0]=0.
        a=angles(origin+future)
        mean_direction=angles(origin+np.einsum('n,nhk->hk',w,future))
        error=angle_error(a,mean_direction)
        spread=np.sqrt(np.einsum('n,nhk->hk',w,error**2))
        features.extend([repeat(metadata[i]),spread])
        means.append(mean_direction)
    for j,k in [(0,1),(1,2),(0,2)]:
        features.append(abs(angle_error(means[j],means[k])))
    return np.column_stack(features)

def vector_features(h,c,rep):
    v=np.diff(c,axis=0)/.5
    diff=np.vstack([v[0]-observation_velocity(h),np.diff(v,axis=0)])
    norm=np.linalg.norm(v,axis=1)
    prior=np.vstack([observation_velocity(h),v[:-1]])
    d=norm*np.linalg.norm(prior,axis=1)
    cos=np.divide((v*prior).sum(1),d,out=np.ones_like(d),where=d>1e-8)
    turn=np.maximum.accumulate(np.arccos(np.clip(cos,-1,1)))
    geom=np.column_stack([v,v-observation_velocity(h),norm,diff/.5,turn])
    j=np.maximum(K-1,0)
    # h=0 has no decoded future step.
    x=np.column_stack([repeat(observation_velocity(h)),geom[j],rep[j]])
    x[0,3:]=0.
    return x

class Advanced:
    def __init__(self,name):
        self.name=name
        self.net=adv.Predictor(name,SEED)
        if name=='RAFT':
            self.bank=DiagnosticBank.__new__(DiagnosticBank)
            self.bank.__dict__.update(vars(self.net.bank))

    def extract(self,histories,special=True):
        if not special:
            return [(c,None) for c,_ in self.net.batch(histories)]
        if self.name=='Chronos-2':
            quantiles=[]
            forecast=adv.forecast
            def capture(q,obs):
                quantiles.append(q)
                return forecast(q,obs)
            with patch.object(adv,'forecast',capture):
                raw=self.net.raw(histories)
            result=[]
            for (c,width,_),q in zip(raw,quantiles,strict=True):
                asym=np.column_stack([q[:,:,1]-q[:,:,0],q[:,:,2]-q[:,:,1]])
                growth=np.vstack([np.zeros(2),np.diff(width,axis=0)/.5])
                x=np.column_stack([np.vstack([np.zeros(6),asym])[K],width[K],
                                   np.maximum.accumulate(width,axis=0)[K],growth[K]])
                result.append((c,x))
            return result
        if self.name=='RAFT':
            with patch.object(self.net,'bank',self.bank):
                raw=self.net.raw(histories)
            return [(c,raft_features(h[-1].position,self.bank.diagnostics,i))
                    for i,(h,(c,_,_)) in enumerate(zip(histories,raw,strict=True))]
        return [(c,vector_features(h,c,rep)) for h,(c,_,rep) in zip(histories,self.net.raw(histories),strict=True)]
