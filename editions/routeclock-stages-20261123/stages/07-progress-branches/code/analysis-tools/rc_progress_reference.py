"""Causal route geometry at a forecast origin. Distances are in metres."""
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
GRID=np.arange(0.,122.,2.)
HS=np.arange(0.,10.1,.5)
def log(msg):
    line=f"{datetime.now():%F %T} RC进度与分支 {msg}"
    with (ROOT.parent/'实时实验进度.txt').open('a') as f:print(line,file=f)
    print(line,flush=True)

def unit(x):
    return x/max(np.linalg.norm(x),1e-8)

def sample(points, s, closed=False):
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if closed and arc[-1]>1e-6:s=np.mod(s,arc[-1])
    q=np.stack([np.interp(s,arc,p) for p in points.T],axis=-1)
    return q+np.maximum(s-arc[-1],0)[:,None]*unit(points[-1]-points[-2])

def pack(hist,mem,imm,p):
    end=hist[-1]; obs=hist[-181:]
    tt=np.array([o.t-end.t for o in obs]); xyz=np.array([o.position for o in obs])
    # Real, irregular events over the last 32 seconds, interpolated only within observed history.
    lag=np.arange(-32.,1.)
    histpos=np.stack([np.interp(lag,tt,col) for col in xyz.T],axis=1)
    age=lag-tt[np.maximum(np.searchsorted(tt,lag,side='right')-1,0)]
    seq=np.column_stack([np.diff(histpos,axis=0),np.clip(age[1:],-32,32),lag[1:]>=tt[0]])
    state=imm.state()[0]; velocity=state[3:]; speed=np.linalg.norm(velocity)
    rc=p.centers['RC']; weights=p.diagnostics['gate_weights']['RC']
    curves=[];meta=[];valid=[]; ref=end.position
    def add(q,feat,ok):
        curves.append(q-q[0]);meta.append(feat);valid.append(ok)
    frag=mem.fragments[mem.active] if mem.active is not None else None
    d0=0 if frag is None else max(0.,frag.arc()[-1]-mem.s)
    active=ref+GRID[:,None]*unit(velocity) if frag is None else sample(frag.points,mem.s+GRID,frag.closed)
    add(active,[d0,0,0,0,int(frag.closed) if frag else 0],frag is not None)
    connections=sorted([(n,d,e) for (src,d),(n,e) in mem.connections.items() if src==mem.active and frag and not frag.closed],reverse=True)
    for i in range(2):
        if i>=len(connections):
            add(active,[0]*5,False);continue
        count,dst,entry=connections[i]; nxt=mem.fragments[dst]
        start=sample(nxt.points,np.array([entry]),nxt.closed)[0]
        terminal=frag.points[-1];bridge=np.linalg.norm(start-terminal)
        rem=GRID-d0
        q=np.where((rem<=0)[:,None],active,np.where((rem<bridge)[:,None],terminal+rem[:,None]*unit(start-terminal),sample(nxt.points,entry+np.maximum(rem-bridge,0),nxt.closed)))
        add(q,[d0,bridge,count,np.dot(unit(nxt.points[-1]-start),unit(velocity)),int(nxt.closed)],True)
    # A past traversal supplies an observed branch at crossings before a fragment terminal.
    # Need 10 seconds of continuation already observed and exclude the current traversal.
    candidates=np.flatnonzero((tt<=-12)&(tt>=tt[0]+3))
    best=None
    for j in candidates:
        k=np.searchsorted(tt,tt[j]-3)
        direction=unit(xyz[j]-xyz[k]); err=np.linalg.norm((histpos[-1]-histpos[-4])-(xyz[j]-xyz[k]))
        score=np.linalg.norm(xyz[j]-ref)+.6*err+2*(1-np.dot(direction,unit(velocity)))
        if best is None or score<best[0]:best=(score,j)
    if best is None:add(active,[0]*5,False)
    else:
        score,j=best; pts=xyz[j:np.searchsorted(tt,tt[j]+12,side='right')]
        if len(pts)<2:add(active,[0]*5,False)
        else:add(sample(pts,GRID),[score,-tt[j],np.linalg.norm(xyz[j]-ref),np.linalg.norm(pts[-1]-pts[0]),0],True)
    prog=mem.progress
    v0=speed if prog is None else max(0.,float(prog.weights@prog.x[:,1]))
    speed_prior=np.clip(v0,0,12)
    x=np.r_[state[3:],speed_prior,mem.innovation,mem.heading_error,mem.entropy,mem.revisits,d0,
            len(connections),len(hist),np.diag(end.covariance),imm.nis,imm.weights]
    return {k:np.asarray(v,dtype='float32') for k,v in dict(seq=seq,x=x,curves=curves,meta=meta,valid=valid,base=rc,pathbase=p.centers['P'],gate=weights,v0=speed_prior,origin=ref).items()}
