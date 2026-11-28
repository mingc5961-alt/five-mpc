"""Run frozen policies and predictors with real motion and the original radio ledger."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np

from field_worlds import OUT,world
from history import put
from policy_env import initialize,plan_events
from predictors import Model
from learned_matrix import policy
from slots import compute_slot

# P compares recipients with fixed purchase times; Q compares timing and recipients.
ROWS={'P01':('fixed_due','Acal'),'P02':('fixed','Acal'),'Q01':('due','Acal'),
      'Q02':('due','M'),'Q05':('due','Fstar'),'Q03':('joint','Acal'),'Q04':('joint','M'),
      'Q07':('joint','RAFT'),'Q08':('joint','Chronos'),'C_due':('due','Chronos')}
SEEDS=(17,29,41)
SCALE=.75

def event_hash(events):
    canonical=[(int(user),round(t*1e6)) for user,t in events]
    return hashlib.sha256(json.dumps(canonical,separators=(',',':')).encode()).hexdigest()

def shorter(w,clock,forecast,factors,grid,stream):
    import paid_full as pf
    radio=pf.Radio(w.cfg);radio.threshold-=5
    horizon=grid.service_us/1e6;cost=grid.fix_us/1e6;rho=grid.budget_us/grid.service_us
    ledger=pf.DeferredLedger(w.cfg,w,radio,0.,horizon);bucket=pf.TokenBucket(cost,rho)
    slots=[SimpleNamespace(origin=0.,action=None,center=None,front=[]) for _ in range(w.n)]
    fixed=set(grid.fixed_indices());events=[];now=0.
    requested={} if stream is None else {t:u for u,t,e in stream}
    for tick in range(grid.opportunities):
        us=grid.start_us(tick);start=us/1e6
        due=[s.origin+(s.action.interval-cost if s.action else 0.) for s in slots]
        if stream is not None:
            if us not in requested:continue
            u=requested[us]
        else:
            if clock=='fixed_due' and tick not in fixed:continue
            u=tick if tick<w.n else int(np.argmin(due))
            if clock=='due' and tick>=w.n:
                wait,sense=grid.allowed(tick,len(events))
                if not sense or (wait and start+1e-12<due[u]):continue
        end=start+cost
        if start+1e-10<bucket.ready(now):raise ValueError('token rate')
        ledger.advance(now,start,slots,False);bucket.consume(start);ledger.advance(start,end,slots,True)
        c,width=forecast(u,w.observe(u,end));s=slots[u];s.origin=end;s.center=c
        s.front=pf.efficient_frontier(radio,c,width,SCALE*factors)
        front=[[] if v is not s and (v.action is None or v.origin+v.action.interval<=end) else
               [a for a in v.front if v.origin+a.interval>end] for v in slots]
        for v,action in zip(slots,pf.choose_actions(front,cost,rho),strict=True):v.action=action
        now=end;events.append([u,end])
    ledger.advance(now,horizon,slots,False)
    return events,ledger.result(cost,rho)

@lru_cache(maxsize=2)
def load_panel(source,panel):
    specs=json.loads((OUT/(source+'_inputs.json')).read_bytes())['panels']
    d=next(p for p in specs if p['id']==panel)
    return d,*world(d)

def work(job):
    from closed import simulate_controls
    from aoi_reference import QuotaGrid
    from panel_types import angle_deg
    import paid_full as pf
    source,panel,row,seed=job;dest=OUT/'receipts'/f'{panel}_{row}_{seed}.json'
    if dest.exists():return json.loads(dest.read_bytes())
    with compute_slot():
        d,w,wh=load_panel(source,panel);duration=d['duration'];clock,kind=ROWS[row]
        grid=QuotaGrid(service_us=duration*10**6,budget_us=round(duration/60*1e6),quota=d['quota'])
        stream=None;weight_sha=None
        if seed is not None:
            net,weight_sha=policy(clock,seed)
            stream,_=plan_events(w,grid,clock,net)
        model=Model(w,kind);angles=[[] for _ in range(16)]
        def forecast(u,obs):
            center,width=model.predict(u,obs)
            if obs.t+2<=duration:
                xyz,_=w.truth(u,obs.t+2)
                angles[u].append(float(angle_deg(center[4],xyz)))
            return center,width
        if duration==180:
            name='stream' if stream is not None else clock
            events,result,_=simulate_controls(w,name,forecast,model.factors,pf,3,SCALE,grid.quota,event_stream=stream)
        else:
            events,result=shorter(w,clock,forecast,model.factors,grid,stream)
        if stream is not None and [(u,round(t*1e6)) for u,t in events]!=[(u,e) for u,_,e in stream]:raise ValueError('stream')
        if len(events)!=grid.quota or [u for u,t in events[:16]]!=list(range(16)):raise ValueError('quota')
        if abs(result['ledger_s'][0]-grid.quota*.0031)>1e-7 or abs(sum(result['ledger_s'])-duration)>1e-6:raise ValueError('airtime')
        if result['probes_per_user']!=round(duration/.02) or abs(sum(result['bits'])/(duration*1e6)-result['G_mbps'])>1e-7:raise ValueError('delivered bits')
        # At 180 s verify that the separate horizon-aware implementation recovers the locked original.
        if row=='Q01' and duration==180:
            same=Model(w,kind);ev,r=shorter(w,clock,same.predict,same.factors,grid,stream)
            if ev!=events or r!=result:raise ValueError('transport parity against locked 180 s ledger')
        rr=dict(source=source,panel=panel,view=d['view'],duration=duration,cohort=d['cohort'],row=row,
                seed=seed,model=kind,scale=SCALE,quota=grid.quota,physics=wh,policy_sha256=weight_sha,
                paid_event_sha256=event_hash(events),events=events,result=result,
                angle_2s_per_user=[float(np.mean(a)) if a else None for a in angles],
                angle_2s_counts=[len(a) for a in angles])
        put(dest,rr);return rr

def main(source):
    initialize();(OUT/'receipts').mkdir(exist_ok=True)
    designs=json.loads((OUT/(source+'_inputs.json')).read_bytes())['panels']
    jobs=[(source,d['id'],r,s) for d in designs for r,(clock,k) in ROWS.items()
          for s in (SEEDS if clock in ('fixed','joint') else (None,))]
    jobs.sort(key=lambda j:ROWS[j[2]][1]!='Chronos')
    put(OUT/(source+'_queue.json'),dict(jobs=jobs,scale=SCALE))
    with ProcessPoolExecutor(min(32,len(jobs))) as pool:
        results=[f.result() for f in as_completed([pool.submit(work,j) for j in jobs])]
    for d in designs:
        rows=[r for r in results if r['panel']==d['id']]
        assert len({r['physics'] for r in rows})==1
        fixed={tuple(round(t*1e6) for _,t in r['events']) for r in rows if r['row'] in ('P01','P02')}
        assert len(fixed)==1
        for seed in SEEDS:
            assert len({r['paid_event_sha256'] for r in rows if r['seed']==seed and r['row'].startswith('Q')})==1
    put(OUT/(source+'_done.json'),dict(checked_endpoints=len(results),panels=len(designs),
        spectrum='simulated 28 GHz; actual motion and timestamps, asynchronously assembled fleet'))
if __name__=='__main__':
    import sys
    main(sys.argv[1])
