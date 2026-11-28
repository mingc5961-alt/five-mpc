"""Paid-state decisions and additive rewards taken from the original deferred radio ledger."""
import sys
import numpy as np
from history import ROOT,ASSETS,RUNTIME,configure
from paired_grid import load_checked,BASE
from paired_models import Predictor

def initialize():
    configure(ASSETS,RUNTIME)
    import grurun,torch
    torch.set_num_threads(1)
    sys.path.append(str(ROOT.parent/'g1-server-handoff/science'))

def trace(ledger,grid):
    """Attribute original probe failures and actually delivered bits to each opportunity window."""
    from refine.network import HS,interpolate
    from routeclock.geometry import angles
    from routeclock.radio import array_gain
    times=ledger.probes;steps=grid.opportunities
    ticks=np.asarray([grid.start_us(i)/1e6 for i in range(steps)])
    bucket=lambda sec:np.searchsorted(ticks,sec,side='right')-1
    fail=np.zeros(steps,dtype=np.int64);bits=np.zeros(steps)
    state=np.asarray(ledger.states);centers=np.asarray(ledger.centers)
    pe=np.searchsorted(ledger.ends,times,side='right')
    ep=np.concatenate([np.full(len(u),i) for i,u,_,_ in ledger.slots])
    users,starts,ends=(np.concatenate([s[j] for s in ledger.slots]) for j in (1,2,3))
    for uid in range(ledger.world.n):
        ref=state[:,uid];expiry=ref[:,0]+ref[:,2]
        good=(ref[pe,2]>0)&(times<expiry[pe])
        fail+=np.bincount(bucket(times[~good]),minlength=steps)
        sel=users==uid;e,left,right=ep[sel],starts[sel],ends[sel]
        stop=np.where(ref[e,2]>0,np.maximum(left,np.minimum(right,expiry[e])),left)
        ok=stop>left;l,r=left[ok],stop[ok];mid=(l+r)*.5
        both=np.r_[times[good],mid];epoch=np.r_[pe[good],e[ok]];n=int(good.sum())
        if not len(both):continue
        keys=ref[epoch,1].astype(int);age=both-ref[epoch,0];pred=np.empty((len(both),3))
        order=np.argsort(keys,kind='stable')
        for ix in np.split(order,np.flatnonzero(np.diff(keys[order]))+1):
            pred[ix]=np.column_stack([np.interp(age[ix],HS,v) for v in centers[keys[ix[0]]].T])
        pointing=angles(pred);truth=interpolate(ledger.world.positions[uid],both,ledger.world.dt)
        pairs=ref[epoch,3:].astype(int);gain=array_gain(pointing,angles(truth),pairs.T)
        loss=10*np.log10(np.prod(pairs,axis=1)/gain)
        fail+=np.bincount(bucket(times[good][loss[:n]>3]),minlength=steps)
        if not len(mid):continue
        pair=pairs[n:].T
        nominal=ledger.radio.snr(pred[n:],pointing[n:],pair)
        mcs=np.searchsorted(ledger.radio.threshold,nominal,side='right').clip(1,len(ledger.radio.se))-1
        shadow=interpolate(ledger.world.shadows[uid],mid,ledger.world.dt)
        snr=ledger.radio.snr(truth[n:],pointing[n:],pair,shadow)
        rate,_=ledger.radio.goodput(snr,mcs)
        i=bucket(l);j=bucket(np.nextafter(r,-np.inf))
        if (j-i>1).any():raise ValueError('a radio slot spanned multiple acquisition windows')
        last=np.minimum(r,np.where(i+1<steps,ticks[np.minimum(i+1,steps-1)],180.))
        bits+=np.bincount(i,weights=rate*(last-l),minlength=steps)
        bits+=np.bincount(j,weights=rate*(r-last),minlength=steps)
    if fail.sum()!=ledger.risk_count.sum() or abs(bits.sum()-ledger.bits.sum())>1:
        raise ValueError(f'nonadditive reward: risk {fail.sum()} vs {ledger.risk_count.sum()}, bits error {bits.sum()-ledger.bits.sum()}')
    return bits,fail

def plan_events(w,grid,protocol,policy,rng=None,epsilon=0.,collect=False):
    import torch
    from aoi_policy import PaidState,choices
    state=PaidState(grid,protocol);events=[];records=[]
    with torch.inference_mode():
        while state.tick<grid.opportunities:
            users,public,mask=state.view();eligible=np.flatnonzero(mask)
            if len(eligible)==1:action=int(eligible[0])
            elif epsilon and rng.random()<epsilon:action=int(rng.choice(eligible))
            else:
                q=policy(torch.from_numpy(users)[None],torch.from_numpy(public)[None])
                action=int(choices(q,torch.from_numpy(mask)[None])[0])
            if collect:records.append((users,public,mask,action,state.queue))
            def observe(uid,end):
                start=grid.start_us(state.tick)/1e6
                return w.observe(uid,start+grid.fix_us/1e6)
            end=state.step(action,observe,mask)
            if end is not None:events.append((action,end-grid.fix_us,end))
    if len(events)!=grid.quota or [u for u,_,_ in events[:16]]!=list(range(16)):raise ValueError('broken paid stream')
    return events,records

def episode(split,k,kind,protocol,policy,budget,quota,multiplier,rng=None,epsilon=0.,collect=False):
    import paid_full as pf
    from aoi_reference import QuotaGrid
    from closed import simulate_controls
    w,meta=load_checked(split,k);w.cfg['simulation']['sensing_budget']=budget/180
    grid=QuotaGrid(budget_us=round(budget*1e6),quota=quota)
    events,records=plan_events(w,grid,protocol,policy,rng,epsilon,collect)
    model=Predictor(w,kind);capture=[]
    def factory(*a):
        ledger=pf.DeferredLedger(*a);capture.append(ledger);return ledger
    observed,result,_=simulate_controls(w,'stream',model.predict,model.factors,pf,budget,multiplier,quota,event_stream=events,ledger_factory=factory)
    if [(u,round(t*1e6)) for u,t in observed]!=[(u,end) for u,_,end in events]:raise ValueError('event stream replay differs')
    payload=None
    if collect:
        bits,fail=trace(capture[0],grid);u,p,mask,action,queue=map(np.asarray,zip(*records))
        payload=dict(users=u,public=p,legal=mask,action=action,queue=queue,bits=bits,fail=fail)
    return result,events,payload
