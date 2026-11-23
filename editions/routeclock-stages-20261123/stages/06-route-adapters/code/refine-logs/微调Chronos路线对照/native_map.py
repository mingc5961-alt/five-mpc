import ctypes as C
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from routeclock.memory import RouteMemory
from routeclock.filters import moment
UNACCELERATED=RouteMemory.forecast
LIB=C.CDLL(str(Path(__file__).with_name('map_lookup.so')))
I=C.c_int;V=C.c_void_p
LIB.lookup.argtypes=[I,V,V,V,V,V,I,V,V,V,V,V,V,I,V,V,C.POINTER(V)]
LIB.lookup.restype=I;LIB.release.argtypes=[V];LIB.release.restype=None
def forecast(mem,hs,point=False):
    if mem.progress is None or point:return UNACCELERATED(mem,hs,point)
    states=mem.progress.forecast(hs);w,x,P=map(np.stack,zip(*states,strict=True))
    matches=mem.matches or [(mem.active,mem.s,1.)]
    nf=len(mem.fragments)
    pts=np.ascontiguousarray(np.concatenate([f.points for f in mem.fragments]))
    arcs=[f.arc() for f in mem.fragments];arc=np.concatenate(arcs)
    beg=np.asarray(np.r_[0,np.cumsum([len(f.points) for f in mem.fragments])],np.int32)
    tangent=np.zeros_like(pts)
    for j,f in enumerate(mem.fragments):
        d=np.diff(f.points,axis=0);tangent[beg[j]:beg[j+1]-1]=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
    closed=np.array([f.closed for f in mem.fragments],np.int32)
    conn=list(mem.connections.items());src=[];dst=[];n=[];entries=[];bridges=[];targets=[]
    for (a,b),(count,entry) in conn:
        d=mem.fragments[b].at(entry)[0]
        src.append(a);dst.append(b);n.append(count);entries.append(entry);targets.append(d);bridges.append(np.linalg.norm(d-mem.fragments[a].points[-1]))
    base=float(mem.progress.weights@mem.progress.x[:,0])
    offset=np.sqrt(3*np.maximum(P[:,:,0,0],0))
    ds=np.stack([np.zeros_like(offset),-offset,offset],axis=-1)
    coords=[];idx=[];mass=[]
    for fi,s,wm in matches:
        length=arcs[fi][-1]
        lifted=s+round((base-s)/length)*length if fi==mem.active and closed[fi] else s
        loc=(x[:,:,0,None]+ds)+(lifted-base);coords.append(loc);idx.append(np.full(loc.shape,fi,dtype=np.int32))
        mass.append(wm*w[:,:,None]*np.array([2/3,1/6,1/6]))
    coord=np.ascontiguousarray(np.stack(coords,axis=1)).reshape(-1)
    ids=np.ascontiguousarray(np.stack(idx,axis=1)).reshape(-1)
    mass=np.stack(mass,axis=1).reshape(-1)
    edge=[np.asarray(a,np.int32) for a in (src,dst,n)]+[np.asarray(a,np.float64) for a in (entries,bridges,targets)]
    ptr=V()
    count=LIB.lookup(nf,*[a.ctypes.data for a in (pts,arc,tangent,beg,closed)],len(conn),*[a.ctypes.data for a in edge],len(coord),ids.ctypes.data,coord.ctypes.data,C.byref(ptr))
    if count<1:raise RuntimeError('route lookup allocation failed')
    try:
        mat=np.ctypeslib.as_array((C.c_double*(count*6)).from_address(ptr.value)).reshape(-1,6)
        ids=mat[:,5].astype(np.int64);probs=mass[ids]*mat[:,4]
        bounds=np.searchsorted(ids,np.arange(len(hs)+1)*9*len(matches))
        means=[];covs=[];supports=[];std=[]
        for (ww,xx,pp),start,stop in zip(states,bounds[:-1],bounds[1:],strict=True):
            cloud=np.ascontiguousarray(mat[start:stop,:3]);weights=probs[start:stop];weights=weights/weights.sum()
            mean=weights@cloud;diff=cloud-mean
            means.append(mean);covs.append(np.einsum('k,ki,kj->ij',weights,diff,diff)+mem.last.covariance)
            supports.append(weights@np.ascontiguousarray(mat[start:stop,3]))
            std.append(np.sqrt(max(moment(ww,xx,pp)[1][0,0],0)))
        return np.stack(means),np.stack(covs),np.array(supports),np.array(std),w
    finally:LIB.release(ptr)
@contextmanager
def native_forecast():
    with patch.object(RouteMemory,'forecast',forecast):yield
if __name__=='__main__':
    from replay_route import setup,get_world,runtime,HS,OUT,write
    setup();worst=0.;ncase=0
    for w in range(40):
        for tr in runtime.warm_trackers(get_world(w)):
            orig=UNACCELERATED(tr.memory,HS)
            alt=forecast(tr.memory,HS)
            for a,b in zip(orig,alt,strict=True):
                worst=max(worst,float(abs(a-b).max()));np.testing.assert_allclose(a,b,atol=5e-10,rtol=1e-12)
            ncase+=1
    write(OUT/'native_route_validation.json',dict(n=ncase,max_abs_difference=worst))
