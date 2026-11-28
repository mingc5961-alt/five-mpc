"""Original native predictors and matched acquisition controls on the frozen Linux bank."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,multiprocessing,os,sys,time
from pathlib import Path
import numpy as np
from history import ROOT,ASSETS,RUNTIME,PACKAGE,configure,check_models,put
from paired_grid import BASE,load_checked,SCALES
from physical_inputs import sha_file
from slots import compute_slot
OUT=BASE/'native_matrix'
CLOCKS={'N01':'native','N02':'native','N03':'pm','N04':'3s','N05':'native','N06':'free','Q09':'due'}
def jobs():
    return [(k,row,b,m) for k in range(1400,1408) for m in SCALES for row,b in
            [('N05',3),('N01',4.5),('N02',3),('N03',3),('N04',3),('N06',3),('Q09',3),('N01',9)]]
def job_key(job):
    k,row,b,m=job
    return f'{row}_{k}_B{b:g}_m{m:.2f}'

def simulate(job):
    import grurun,paid_full
    from closed import simulate_controls
    k,row,b,m=job;key=job_key(job)
    dest=OUT/'receipts'/(key+'.json'); mark=OUT/'active'/dest.name
    w,wm=load_checked('development',k)
    frozen=json.loads((BASE/'summary.json').read_bytes())
    if w.mission!=k or wm['physical_sha256']!=frozen['physical_sha256'][str(k)]:raise ValueError(key)
    if dest.exists():
        r=json.loads(dest.read_bytes())
        if r['key']!=key or r['world']!=wm['physical_sha256'] or r['freeze_sha256']!=frozen['freeze_sha256']:raise ValueError(key)
        return key
    with compute_slot():
        start=time.time();put(mark,dict(pid=os.getpid(),started=start))
        try:
            w.cfg['simulation']['sensing_budget']=b/180.
            pred=grurun.Predictor(w.cfg,'C' if row=='N05' else 'G1')
            if row=='N02':
                pred.arm='Static'
                fixed=np.clip(pred.net.prior+np.interp(grurun.GRID,[0,1,2,5,10],[0,0,0,.05,.10]),0,1)
                np.testing.assert_allclose(fixed[[2,4,10,20]],[.25,.25,.4,.6],rtol=0,atol=1e-6)
            h=[[] for _ in range(w.n)]
            def predict(u,o):h[u].append(o);return pred.update(u,h[u])
            quota=frozen['K'] if row=='Q09' else None
            period=frozen['PM_ms'] if row=='N03' else None
            events,r,forced=simulate_controls(w,CLOCKS[row],predict,pred.factors,paid_full,b,m,quota,period)
            if row=='N04' and len(events)!=960:raise ValueError('3-second clock')
            if row=='Q09' and len(events)!=quota:raise ValueError('quota')
            put(dest,dict(key=key,protocol='g1_linux_paired_v2',split='development',world_id=k,row=row,budget_s=b,
                scale=m,world=wm['physical_sha256'],freeze_sha256=frozen['freeze_sha256'],cost_s=.0031,events=events,result=r,
                period_ms=period,quota=quota,forced=forced,elapsed_s=time.time()-start))
        finally:mark.unlink(missing_ok=True)
    return key

def main():
    check_models(PACKAGE/'reference',ASSETS,with_chronos=True)
    configure(ASSETS,RUNTIME)
    import grurun,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.append(str(ROOT.parent/'g1-server-handoff/science'))
    from closed import freeze,quota_period
    for sub in ('receipts','active'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    tasks=jobs();start=time.time()
    put(OUT/'batch.json',dict(start=start,total=len(tasks),workers=32,pid=os.getpid(),phase='development'))
    ctx=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(16,mp_context=ctx) as heavy,ProcessPoolExecutor(16,mp_context=ctx) as light:
        pending=[(heavy if j[1]=='N05' else light).submit(simulate,j) for j in tasks]
        for f in as_completed(pending):f.result()
    summaries={}
    for row,b in dict.fromkeys((j[1],j[2]) for j in tasks):
        by={m:[json.loads((OUT/'receipts'/(job_key((k,row,b,m))+'.json')).read_bytes()) for k in range(1400,1408)] for m in SCALES}
        policy=freeze(by);item=dict(policy=policy)
        if row=='N01' and policy['feasible']:item['K'],item['PM_ms']=quota_period(by,policy,b)
        summaries[f'{row}_B{b:g}']=item
    put(OUT/'summary.json',dict(completed=len(tasks),development_only=True,wall_s=time.time()-start,rows=summaries))
    put(OUT/'batch.json',dict(start=start,total=len(tasks),workers=32,pid=os.getpid(),phase='done'))
if __name__=='__main__':main()
