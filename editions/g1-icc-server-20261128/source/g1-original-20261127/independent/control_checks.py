"""Acceptance of clock and ledger combinations using original G1 on historical physics."""
import concurrent.futures as futures
import json
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path

from history import ASSETS, PACKAGE, ROOT, RUNTIME, configure, check_models, put

OUT=ROOT/'recovery_01/clock_diagnostics'
SOURCE=ROOT.parent/'g1-server-handoff'
WORLD=ROOT/'recovery_01/anchor_g1/work/factorial_run/confirm0.world'

def key(case):
    clock,b,k,m=case
    return f'{clock}_B{b:g}_K{k}_m{m:.2f}'

def run(case):
    import grurun, paid_full
    from closed import simulate_controls
    from aoi_reference import QuotaGrid
    from worlds import fingerprint
    clock,b,k,m=case
    tag=key(case)
    mark=OUT/'active'/f'{tag}.json'
    put(mark,dict(pid=os.getpid(),started=time.time()))
    begin=time.monotonic()
    try:
        with WORLD.open('rb') as stream:w=pickle.load(stream)
        w.cfg['simulation']['sensing_budget']=b/180.
        engine=grurun.Predictor(w.cfg,'G1')
        histories=[[] for _ in range(w.n)]
        def predict(u,observation):
            histories[u].append(observation)
            return engine.update(u,histories[u])
        events,result,forced=simulate_controls(w,clock,predict,engine.factors,paid_full,b,m,k)
        def require(condition,msg):
            if not condition:raise ValueError(f'{tag}: {msg}')
        us=[(u,round(t*1e6)) for u,t in events]
        require(all(abs(t*1e6-v)<1e-4 for (_,t),(_,v) in zip(events,us)), 'microsecond resolution')
        require(all(a[1]<c[1] for a,c in zip(us,us[1:])),'ordered fixes')
        grid=QuotaGrid(budget_us=round(b*1e6),quota=k or 16)
        fixed=[grid.start_us(i)+grid.fix_us for i in grid.fixed_indices()]
        spacing=grid.start_us(1)
        require(us[:16]==[(u,3100+u*spacing) for u in range(16)],'initial stagger')
        if clock=='3s':
            target=[(u,3_000_000*j+u*spacing+3100) for j in range(60) for u in range(16)]
            require(us==target,'full 3 s clock')
        if clock in ('oldest','fixed_due','due'):
            require(len(us)==k and abs(result['ledger_s'][0]-.0031*k)<1e-7,'exact quota charge')
            require(all((end-3100)%spacing==0 for u,end in us),'off-grid fix')
        if clock in ('oldest','fixed_due'):
            require([t for u,t in us]==fixed,'fixed success times')
        if clock=='oldest':require([u for u,t in us]==[i%16 for i in range(k)],'oldest order')
        receipt=dict(clock=clock,budget_s=b,diagnostic_quota=k,multiplier=m,world=fingerprint(w),
                     events=events,result=result,forced=forced,original_model=True,formal_endpoints=0,
                     elapsed_s=time.monotonic()-begin)
        if clock=='native':
            from audit_receipt import audit
            sample=OUT/'native_candidate.json'
            put(sample,dict(world=receipt['world'],events=events,result=result))
            ref=ROOT/'recovery_01/anchor_g1/run_confirm_0_G1_0.45.json'
            comparison=audit(ref,sample)
            require(comparison['physical_match'] and comparison['actual_events']==comparison['expected_events']==comparison['exact_event_pairs']
                    and max(f['absmax'] for f in comparison['fields'].values())==0., 'native parity with original loop')
        put(OUT/'receipts'/f'{tag}.json',receipt)
        return tag
    finally:mark.unlink(missing_ok=True)

def main():
    configure(ASSETS,RUNTIME)
    check_models(PACKAGE/'reference',ASSETS)
    import grurun, torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.append(str(SOURCE/'science'))
    import closed
    for sub in ['active','receipts']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    cases=[('native',3,None,.45),('3s',3,None,.75)]
    for b,k in [(3,700),(3,950),(4.5,950),(9,950)]:
        cases.extend((clock,b,k,.75) for clock in ('oldest','fixed_due','due'))
    cases.extend([('fixed_due',3,700,.45),('oldest',3,700,.45)])
    start=time.time()
    put(OUT/'batch.json',dict(pid=os.getpid(),start=start,total=len(cases),workers=8,phase='controls'))
    with futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('fork')) as pool:
        for tag in pool.map(run,cases): print(tag,'OK',flush=True)
    def read(case):return json.loads((OUT/'receipts'/f'{key(case)}.json').read_bytes())
    c1=read(('oldest',3,700,.45));c2=read(('oldest',3,700,.75))
    assert c1['events']==c2['events']
    for b,k in [(3,700),(3,950),(4.5,950),(9,950)]:
        t1=[t for u,t in read(('oldest',b,k,.75))['events']]
        t2=[t for u,t in read(('fixed_due',b,k,.75))['events']]
        assert t1==t2
    put(OUT/'summary.json',dict(passed=len(cases),formal_endpoints=0,controls=['native parity','periodic 3s: 960 fixes',
        'oldest/fixed-due identical fixed times', 'exact cost at K=700/950', 'oldest event stream identical across multipliers',
        'wait and advance full ledger at B=3/4.5/9'],wall_s=time.time()-start))
    put(OUT/'batch.json',dict(pid=os.getpid(),start=start,total=len(cases),workers=8,phase='done'))

if __name__=='__main__':main()
