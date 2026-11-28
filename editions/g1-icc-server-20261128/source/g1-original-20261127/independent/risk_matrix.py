"""Independent IMM tracking and an IMM covariance-triggered equal-cost refresh rule."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json,multiprocessing,os,time
from statistics import NormalDist
import numpy as np
from history import put
from paired_grid import BASE,SCALES,load_checked
from predictors import Model
from policy_env import initialize
from slots import compute_slot
OUT=BASE/'risk_matrix'
COVERAGES=(.5,.8,.95)
def evaluate(w,row,b,q,m,coverage):
    import paid_full as pf
    from closed import simulate_controls
    from routeclock.geometry import angular_jacobian
    model=Model(w,'Acal' if row=='Q06' else 'IMM')
    deadlines=[{} for _ in range(w.n)]
    radio=pf.Radio(w.cfg)
    def forecast(u,o):
        center,width=model.predict(u,o)
        if row=='Q06':
            track=model.base.states[u].imm
            projected,cov=track.forecast(pf.GRID)
            j=np.array([angular_jacobian(v) for v in projected])
            sd=np.sqrt(np.maximum(1e-12,np.einsum('hij,hjk,hik->hi',j,cov,j)))
            aim=pf.angle(center)
            envelope=np.abs(pf.difference(pf.angle(projected),aim))+NormalDist().inv_cdf(1-(1-coverage)/4)*sd
            for pair in radio.pairs:
                loss=radio.worst_losses(aim,envelope,pair);ix=np.flatnonzero(loss>3.)
                deadlines[u][tuple(pair)]=o.t+(pf.GRID[ix[0]] if len(ix) else pf.GRID[-1])
        return center,width
    def due(slots):
        return [s.origin if s.action is None else deadlines[u][tuple(s.action.pair)] for u,s in enumerate(slots)]
    return simulate_controls(w,'due',forecast,model.factors,pf,b,m,q,refresh_deadlines=due if row=='Q06' else None)
def work(job):
    k,row,m,coverage,tuning=job;tag=f'{row}_{k}_m{m:.2f}'
    folder=OUT/('tuning' if tuning else 'receipts')
    dest=folder/(f'{tag}_c{coverage:g}.json' if tuning else tag+'.json')
    if dest.exists():return json.loads(dest.read_bytes())
    with compute_slot():
        marker=OUT/'active'/dest.name;put(marker,dict(pid=os.getpid(),started=time.time()))
        try:
            q=json.loads((BASE/'summary.json').read_bytes())['K'];w,meta=load_checked('development',k)
            events,r,forced=evaluate(w,row,3,q,m,coverage)
            if len(events)!=q or abs(r['ledger_s'][0]-q*.0031)>1e-7:raise ValueError('matched cost')
            record=dict(protocol='g1_linux_paired_v2',row=row,world_id=k,world=meta['physical_sha256'],scale=m,
                        quota=q,budget_s=3,coverage=coverage,forced=forced,events=events,result=r)
            put(dest,record);return record
        finally:marker.unlink(missing_ok=True)
def main():
    initialize()
    from closed import freeze
    for f in ('receipts','active','tuning'):(OUT/f).mkdir(parents=True,exist_ok=True)
    put(OUT/'protocol.json',dict(row='Q06',coverages=COVERAGES,physical_threshold_db=3,
        envelope='two-axis normal Bonferroni box plus difference between paid IMM mean and offered Acal pointing; first 0.5s grid crossing for actual chosen array',
        decision='earliest crossing deadline among users; WAIT when not reached unless exact quota requires a purchase',
        selection='common .75: highest mean G subject to R<=.07; else lowest mean R; tie lower coverage',
        threshold_tuning_worlds=list(range(1400,1408))))
    start=time.time();put(OUT/'batch.json',dict(total=64,workers=24,pid=os.getpid(),start=start,phase='tuning'))
    with ProcessPoolExecutor(24,mp_context=multiprocessing.get_context('fork')) as p:
        fits=list(p.map(work,[(k,'Q06',.75,c,True) for c in COVERAGES for k in range(1400,1408)]))
        stats={c:(float(np.mean([r['result']['G_mbps'] for r in fits if r['coverage']==c])),
                  float(np.mean([r['result']['R'] for r in fits if r['coverage']==c]))) for c in COVERAGES}
        def rank(c):
            g,r=stats[c];return (1,g,-r,-c) if r<=.07 else (0,-r,g,-c)
        selected=max(COVERAGES,key=rank)
        put(OUT/'threshold.json',dict(selected=selected,points=stats,development_only=True))
        for r in fits:
            if r['coverage']==selected:put(OUT/'receipts'/f"Q06_{r['world_id']}_m0.75.json",r)
        items=list(p.map(work,[(k,row,m,selected if row=='Q06' else None,False) for row in ('Q06','Q10')
                                 for k in range(1400,1408) for m in SCALES]))
    policies={row:freeze({m:[x for x in items if x['row']==row and x['scale']==m] for m in SCALES}) for row in ('Q06','Q10')}
    put(OUT/'summary.json',dict(count=len(items),policies=policies,threshold=selected,additional_tuning_runs=16))
    put(OUT/'batch.json',dict(total=64,workers=24,pid=os.getpid(),start=start,phase='done'))
if __name__=='__main__':main()
