"""Paired geometric scoring on identical sparse updates, with natural-age denominators."""
import csv
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from server_paths import RUN
import paid_full as pf
from fit_models import tracks, packets
from common import truth
from worlds import load_world
from routeclock.radio import array_gain

AGES=(.5,1.,2.,3.,5.,8.,10.)
COUNTS=(1,2,3,4,6,8,12,24,40)
PRIMARY_N={6,8,12,24,40}
PRIMARY_A={2.,3.,5.}


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


def life(w,u,t,center,width,factors,scale,radio):
    dense=np.arange(501)*.02
    path=np.column_stack([np.interp(dense,pf.GRID,col) for col in center.T])
    real=truth(w,u,t+dense)
    ok=np.logical_and.accumulate(array_gain(pf.angle(path),pf.angle(real),(32,32))/1024>=10**(-.3))
    hs=pf.network.QUERY_H[1:]
    hreal=max([0.,*hs[ok[np.rint(hs*50).astype(int)]]])
    env=np.maximum(width,np.vstack([width[:1],width[:-1]]))
    losses=radio.worst_losses(np.tile(pf.angle(center),(len(hs),1)),(scale*factors[:,None,None]*env).reshape(-1,2),(32,32)).reshape(len(hs),-1)
    accept=np.logical_and.accumulate(np.where(pf.GRID[None,:]<=hs[:,None],losses,0).max(1)<=3)
    return hreal,max([0.,*hs[accept]])


def one(k):
    split='rebuild_holdout'
    dst=RUN/'panel'/f'{k}.csv.gz'
    if dst.exists():return
    dst.parent.mkdir(parents=True,exist_ok=True)
    tracks((split,k))
    w=load_world(split,k)
    geom=[Geometry(w.times,w.positions[u],w.velocities[u]) for u in range(w.n)]
    static=json.loads((RUN/'models/static.json').read_bytes())['weights']
    radio=pf.Radio(w.cfg)
    names=['A','M','Fstar']
    tube={v:pf.bundle(v) for v in names}
    tmp=dst.with_suffix(f'.{os.getpid()}.tmp')
    with np.load(RUN/'tracks'/f'{k}.npz') as a,gzip.open(tmp,'wt',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['world','u','n','t','age','h','natural','label','group','e_A','e_M','e_Fstar',
                         'real_A','adm_A','real_M','adm_M','real_Fstar','adm_Fstar'])
        next_t=np.full(len(a['t']),np.inf);uid=a['u'];times=a['t']
        for u in range(w.n):
            ids=np.flatnonzero(uid==u);next_t[ids[:-1]]=times[ids[1:]]
        for idx,(pk,u,t,n) in enumerate(zip(packets(a),uid,times,a['n'])):
            if n not in COUNTS:continue
            curves=[];limits=[]
            for name in names:
                c,feats=pf.parts(pk,name,static)
                curves.append(c)
                limits.extend(life(w,u,t,c,pf.raw(tube[name]['tree'],feats),tube[name]['final'],.75,radio))
            for age in AGES:
                label=geom[u].label(t,t+age,w.detour_windows[u])
                for h in (0,2):
                    total=age+h
                    if total>10 or t+total>180:continue
                    target=truth(w,u,t+total)
                    errs=[angle_deg(c[int(2*total)],target) for c in curves]
                    writer.writerow([k,u,n,t,age,h,int(next_t[idx]>t+age),label,w.groups[u],*errs,
                                     *[max(0,v-age) for v in limits]])
    tmp.replace(dst)


if __name__=='__main__':
    with ProcessPoolExecutor(8) as pool:
        for i,_ in enumerate(pool.map(one,range(4000,4024)),1):print(i,flush=True)
