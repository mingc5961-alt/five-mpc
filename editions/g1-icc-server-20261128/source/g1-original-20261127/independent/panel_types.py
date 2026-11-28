"""Shared geometric labels and snapshot predictor architecture from the registered analysis."""
import numpy as np
import torch
from torch import nn

def angle_deg(x,y):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(x,y),axis=-1),np.einsum('...i,...i->...',x,y)))

class Geometry:
    def __init__(self,t,xyz,velocity):
        self.t=t;self.xyz=xyz
        self.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=-1))]
        self.speed=np.linalg.norm(velocity,axis=1)
        self.direction=velocity/np.maximum(self.speed[:,None],1e-9)

    def label(self,t0,tq,detours=()):
        if any(start<=tq<=end for start,end in detours):return 'detour'
        if any(end<tq<=end+5 for _,end in detours):return 'rejoin'
        right=min(max(1,np.searchsorted(self.t,tq)),len(self.t)-1)
        left=right-1
        frac=np.clip((tq-self.t[left])/(self.t[right]-self.t[left]),0.,1.)
        speed=(1-frac)*self.speed[left]+frac*self.speed[right]
        if speed<.5:return 'hover'
        direction=(1-frac)*self.direction[left]+frac*self.direction[right]
        direction/=max(np.linalg.norm(direction),1e-12)
        position=(1-frac)*self.xyz[left]+frac*self.xyz[right]
        arc=(1-frac)*self.arc[left]+frac*self.arc[right]
        last=np.searchsorted(self.t,min(t0,tq-10.),side='right')
        if not last or arc<10:return 'first_pass'
        dist=np.linalg.norm(self.xyz[:last]-position,axis=1)
        candidates=np.flatnonzero((dist<=5)&(self.direction[:last]@direction>=np.cos(np.deg2rad(30)))&
                                  (self.arc[:last]>=10)&(self.speed[:last]>=.5))
        def knots(end):
            lo,hi=np.searchsorted(self.arc,[end-10,end])
            return end-self.arc[lo:hi]
        def points(end,offset):
            return np.column_stack([np.interp(end-offset,self.arc,col) for col in self.xyz.T])
        ends=np.array([0.,10.])
        new_knots=knots(arc)
        for i in candidates[np.argsort(dist[candidates])]:
            old=self.arc[i]
            if np.linalg.norm(points(arc,ends)[-1]-points(old,ends)[-1])>5:continue
            # Piecewise-linear paths have a convex separation norm between their
            # combined arc-length knots. Checking each knot certifies the interval.
            offsets=np.r_[ends,new_knots,knots(old)]
            if np.max(np.linalg.norm(points(arc,offsets)-points(old,offsets),axis=1))<=5:
                return 'same_direction_revisit'
        return 'first_pass'

    def sampled_label(self,t0,tq,detours=()):
        if any(start<=tq<=end for start,end in detours):return 'detour'
        if any(end<tq<=end+5 for _,end in detours):return 'rejoin'
        j=min(np.searchsorted(self.t,tq),len(self.t)-1)
        if self.speed[j]<.5:return 'hover'
        last=np.searchsorted(self.t,min(t0,tq-10.),side='right')
        if not last or self.arc[j]<10:return 'first_pass'
        ids=np.arange(0,last,max(1,round(.1/(self.t[1]-self.t[0]))))
        dist=np.linalg.norm(self.xyz[ids]-self.xyz[j],axis=1)
        match=ids[(dist<=5)&(self.direction[ids]@self.direction[j]>=np.cos(np.deg2rad(30)))&
                  (self.arc[ids]>=10)&(self.speed[ids]>=.5)]
        for old in match:
            offsets=np.array([0.,2.5,5.,7.5,10.])
            newp=self.xyz[np.searchsorted(self.arc,self.arc[j]-offsets)]
            oldp=self.xyz[np.searchsorted(self.arc,self.arc[old]-offsets)]
            if np.all(np.linalg.norm(newp-oldp,axis=1)<=5):return 'same_direction_revisit'
        return 'first_pass'

FEATURES=['x_m','y_m','z_m','radial_mps','cov_xx','cov_yy','cov_zz','radial_std','age_s','extra_s']

def features(o,age,h):
    return [*o.position,o.radial_velocity,*np.diag(o.covariance),o.radial_std,age,h]

class Snapshot(nn.Module):
    def __init__(self,age):
        super().__init__()
        self.age=age
        self.register_buffer('mu',torch.zeros(len(FEATURES)))
        self.register_buffer('sigma',torch.ones(len(FEATURES)))
        self.head=nn.Sequential(nn.Linear(10 if age else 9,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,3))

    def forward(self,x):
        z=(x-self.mu)/self.sigma
        if not self.age:z=torch.cat([z[:,:-2],z[:,-1:]],dim=-1)
        base=nn.functional.normalize(x[:,:3])
        return nn.functional.normalize(base+.05*self.head(z))
