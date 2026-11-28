"""Shared 180 s closed-loop scorer; all purchases are charged on completion."""
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from server_paths import RUN,ROOT
import paid_full as pf
from worlds import load_world, fingerprint
from importlib.util import spec_from_file_location,module_from_spec
# Prefer the unmodified shipped integer grid.
import sys
sys.path.append(str(ROOT/'tools'))
from aoi_reference import QuotaGrid

SCALES=(.45,.60,.75,1.)
ROWS={'N01':('A','native'),'N02':('F0','native'),'N03':('A','pm'),'N04':('A','3s'),
      'N06':('A','free'),'P01':('A','fixed_due'),'P03':('A','oldest'),
      'P_M':('M','oldest'),'P_Fstar':('Fstar','oldest'),
      'Q01':('A','due'),'Q02':('M','due'),'Q05':('Fstar','due')}


def destination(split,k,row,b,m,quota=None,period=None):
    return RUN/'receipts'/f'{split}.{k}.{row}.B{b:g}.m{m:.2f}.k{quota}.p{period}.json'


def stamp(source):
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


class Predictor:
    def __init__(self,w,kind):
        self.kind=kind
        self.net=pf.model()
        self.states=[pf.State(w.cfg,self.net) for _ in range(w.n)]
        self.tube=pf.bundle('A' if kind=='F0' else kind)
        self.static=json.loads((RUN/'models/static.json').read_text())['weights'] if kind=='Fstar' else None

    def __call__(self,u,obs):
        pk=self.states[u].update(obs,self.net)['G']
        c,f=pf.parts(pk,self.kind,self.static)
        return c,pf.raw(self.tube['tree'],f)


def simulate_controls(w, clock, pred, factors, runtime, b, m, quota=None, period=None,
                      event_stream=None, ledger_factory=None, refresh_deadlines=None):
    """Score a causal 180 s clock on the supplied world, predictor and radio implementation."""
    cost=.0031;rho=b/180.
    radio=runtime.Radio(w.cfg);radio.threshold-=5
    ledger=(ledger_factory or runtime.DeferredLedger)(w.cfg,w,radio,0.,180.)
    bucket=runtime.TokenBucket(cost,rho)
    slots=[SimpleNamespace(origin=0.,action=None,center=None,front=[]) for _ in range(w.n)]
    events=[]; forced=0
    now=0.
    def due_times():
        return [s.origin+(s.action.interval-cost if s.action else 0.) for s in slots]
    def purchase(u,start):
        nonlocal now
        if start+cost>180.:raise ValueError('event outside service')
        end=start+cost
        ledger.advance(now,start,slots,False)
        bucket.consume(start)
        ledger.advance(start,end,slots,True)
        c,width=pred(u,w.observe(u,end));s=slots[u]
        s.origin=end;s.center=c
        s.front=runtime.efficient_frontier(radio,c,width,m*factors)
        fronts=[[] if v is not s and (v.action is None or v.origin+v.action.interval<=end)
                else [a for a in v.front if v.origin+a.interval>end] for v in slots]
        for v,a in zip(slots,runtime.choose_actions(fronts,cost,rho),strict=True):v.action=a
        now=end;events.append([u,end])
    if clock=='stream':
        if event_stream is None:
            raise ValueError('a fixed event stream is required')
        for u,start_us,end_us in event_stream:
            start=start_us/1e6
            if end_us-start_us!=3100 or start<now-1e-10 or start+1e-10<bucket.ready(now):
                raise ValueError('inconsistent external sensing event')
            purchase(u,start)
    elif clock=='native':
        while now<180:
            u=len(events) if len(events)<w.n else int(np.argmin(due_times()))
            start=max(now,now if len(events)<w.n else due_times()[u],bucket.ready(now))
            if start+cost>180:break
            purchase(u,start)
    elif clock in ('3s','pm'):
        T=3. if clock=='3s' else period/1000.
        for j in range(int(180/T)+1):
            for u in range(w.n):
                start=j*T+u*(cost/rho)
                if start+cost<=180:purchase(u,start)
    else:
        grid=QuotaGrid(budget_us=round(b*1e6),quota=quota or w.n)
        fixed=set(grid.fixed_indices()) if clock in ('fixed_due','oldest') else None
        for i in range(grid.opportunities):
            start=grid.start_us(i)/1e6
            if fixed is not None and i not in fixed:continue
            if i<w.n:
                u=i
            else:
                due=refresh_deadlines(slots) if refresh_deadlines is not None else due_times()
                u=int(np.argmin([s.origin for s in slots] if clock=='oldest' else due))
                if fixed is None:
                    if clock=='free':
                        if start+1e-12<due[u]:continue
                    else:
                        wait,sense=grid.allowed(i,len(events))
                        if not sense or (wait and start+1e-12<due[u]):continue
                        forced+=int(not wait)
            purchase(u,start)
    ledger.advance(now,180.,slots,False)
    r=ledger.result(cost,rho)
    assert len(events)>=16 and [u for u,_ in events[:16]]==list(range(16))
    assert r['probes_per_user']==9000 and abs(sum(r['ledger_s'])-180)<1e-6
    assert abs(r['ledger_s'][0]-cost*len(events))<1e-7 and r['ledger_s'][0]<=b+1e-7
    assert abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-6
    return events, r, forced


def score(split,k,row,b,m,quota=None,period=None):
    out=destination(split,k,row,b,m,quota,period)
    if out.exists():return json.loads(out.read_bytes())
    started=time.monotonic()
    w=load_world(split,k);kind,clock=ROWS[row]
    pred=Predictor(w,kind)
    events,r,forced=simulate_controls(w,clock,pred,pred.tube['final'],pf,b,m,quota,period)
    if row[0] in 'PQ':assert len(events)==quota
    if row=='N04' and b==3:assert len(events)==960
    rec=dict(protocol='linux_refit_v1',split=split,k=k,row=row,budget_s=b,scale=m,quota=quota,
             period_ms=period,world=fingerprint(w),elapsed_s=time.monotonic()-started,
             events=events,result=r,forced_after_initial=forced,
             event_sha256=hashlib.sha256(json.dumps(events,separators=(',',':')).encode()).hexdigest(),
             model_sha256=stamp(RUN/'models/short_refit.pt'),
             tube_sha256=stamp(RUN/'models'/('tube_'+('A' if kind=='F0' else kind)+'.pkl')))
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(rec,allow_nan=False));temp.replace(out)
    return rec


def freeze(receipts):
    pts={}
    for m,rr in receipts.items():
        pts[m]=np.mean([[r['result']['G_mbps'],100*r['result']['R']] for r in rr],axis=0)
    choices=[(g,-risk,1,{m:1.}) for m,(g,risk) in pts.items() if risk<=7.]
    for i,m in enumerate(pts):
        g,r=pts[m]
        for n in list(pts)[i+1:]:
            v,s=pts[n]
            if (r-7)*(s-7)>=0:continue
            p=(7-s)/(r-s)
            choices.append((p*g+(1-p)*v,-7.,0,{m:p,n:1-p}))
    if choices:
        best=max(choices,key=lambda v:v[:3]);weights=best[-1];feasible=True
    else:
        best=min(pts,key=lambda m:(pts[m][1],-pts[m][0]));weights={best:1.};feasible=False
    return dict(feasible=feasible,weights=weights,points={m:p.tolist() for m,p in pts.items()})


def quota_period(by_scale,policy,b):
    num=sum(p*np.mean([len(r['events']) for r in by_scale[m]]) for m,p in policy['weights'].items())
    quota=int(np.floor(num+.5))
    step=round(.0031/(b/180)*1e6)
    low=3000 if b==3 else int(np.ceil(16*step/1000))
    periods=np.arange(low,max(6001,low+int(2*180000*16/num)))
    count=(1+(180000000-3100-np.arange(16)*step)//(periods[:,None]*1000)).sum(1)
    diff=abs(count-num)
    tied=periods[diff==diff.min()]
    return quota,int(tied[(len(tied)-1)//2])
