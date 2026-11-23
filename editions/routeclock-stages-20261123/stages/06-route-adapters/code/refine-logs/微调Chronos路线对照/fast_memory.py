"""Vectorize independent map lookups; preserve branching for learned fragment connections."""
import numpy as np
from contextlib import contextmanager
from unittest.mock import patch
from routeclock.memory import RouteMemory,Fragment
from routeclock.filters import moment

def at_many(fragment,arc,coordinates):
    s=np.asarray(coordinates)
    if fragment.closed and arc[-1]>0:s=s%arc[-1]
    j=(np.searchsorted(arc,s,side='right')-1).clip(0,len(arc)-2)
    d=fragment.points[j+1]-fragment.points[j]
    tangent=d/np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-12)
    supported=(s>=0)&(s<=arc[-1])
    positions=fragment.points[j]+(s-arc[j])[...,None]*tangent
    after=s>arc[-1]
    if np.any(after):positions[after]=fragment.points[-1]+(s[after]-arc[-1])[:,None]*tangent[after]
    return positions,supported.astype(float)

def forecast(memory,hs,point=False):
    if memory.progress is None or point:
        return ORIGINAL(memory,hs,point)
    w,x,P=map(np.stack,zip(*memory.progress.forecast(hs),strict=True))
    base=float(memory.progress.weights@memory.progress.x[:,0])
    matches=memory.matches or [(memory.active,memory.s,1.)]
    offset=np.sqrt(3*np.maximum(P[:,:,0,0],0))
    zero=np.zeros_like(offset)
    ds=np.stack([zero,-offset,offset],axis=-1)
    wp=np.array([2/3,1/6,1/6])
    poses=[];weights=[];supports=[];branch=[]
    for fi,s,wm in matches:
        fragment=memory.fragments[fi];arc=fragment.arc()
        lifted=s+round((base-s)/arc[-1])*arc[-1] if fi==memory.active and fragment.closed else s
        loc=(x[:,:,0,None]+ds)+(lifted-base)
        pos,supp=at_many(fragment,arc,loc)
        mass=(wm*w[:,:,None])*wp
        poses.append(pos);weights.append(mass);supports.append(supp)
        outgoing=any(src==fi for src,dst in memory.connections)
        branch.append((loc>arc[-1]) if outgoing and not fragment.closed else np.zeros_like(loc,dtype=bool))
    xyz=np.stack(poses,axis=1).reshape(len(hs),-1,3)
    mass=np.stack(weights,axis=1).reshape(len(hs),-1)
    support=np.stack(supports,axis=1).reshape(len(hs),-1)
    branched=np.stack(branch,axis=1).reshape(len(hs),-1).any(1)
    # Rare paths beyond a connected terminal must retain every branch and its covariance.
    means=[];covs=[];cert=[];stds=[]
    for j in range(len(hs)):
        _,cov=moment(w[j],x[j],P[j]);stds.append(np.sqrt(max(cov[0,0],0)))
        if branched[j]:
            proj=[];ww=[];ss=[]
            for fi,s,wm in matches:
                frag=memory.fragments[fi];L=frag.arc()[-1]
                lifted=s+round((base-s)/L)*L if fi==memory.active and frag.closed else s
                for mode in range(3):
                    for k in range(3):
                        at=x[j,mode,0]+ds[j,mode,k]+(lifted-base)
                        for pp,supported,weight in memory.map_distribution(fi,at):
                            proj.append(pp);ww.append(wm*w[j,mode]*wp[k]*weight);ss.append(supported)
            cloud=np.array(proj);ww=np.array(ww);ss=np.array(ss)
        else:cloud=xyz[j];ww=mass[j];ss=support[j]
        ww=ww/ww.sum();mu=ww@cloud;d=cloud-mu
        means.append(mu);covs.append(np.einsum('k,ki,kj->ij',ww,d,d)+memory.last.covariance);cert.append(ww@ss)
    return np.array(means),np.array(covs),np.array(cert),np.array(stds),w
ORIGINAL=RouteMemory.forecast

@contextmanager
def vectorized_forecast():
    with patch.object(RouteMemory,'forecast',forecast):yield

if __name__=='__main__':
    from replay_route import setup,runtime,get_world,memoized_arc,HS,write,OUT
    setup();largest=0.;n=0
    for w in range(40):
        for tr in runtime.warm_trackers(get_world(w)):
            with memoized_arc():ref=tr.memory.forecast(HS)
            cand=forecast(tr.memory,HS)
            for a,b in zip(ref,cand,strict=True):
                largest=max(largest,float(abs(a-b).max()))
                np.testing.assert_allclose(a,b,atol=5e-10,rtol=1e-12)
            n+=1
    write(OUT/'route_vectorization.json',dict(n=n,max_abs_difference=largest))
