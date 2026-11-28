"""Replay a policy-selected paid stream across predictors, multipliers and budgets."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from filelock import FileLock
from functools import lru_cache
import hashlib,json,multiprocessing,os,time
from pathlib import Path
import numpy as np
import torch
from history import put
from paired_grid import BASE,load_checked,SCALES
from physical_inputs import sha_file
from policy_env import initialize,plan_events
from deterministic_matrix import budgets
from predictors import Model
from slots import compute_slot
OUT=BASE/'learned_matrix'
ROWS={'P02':('fixed','Acal'),'Q03':('joint','Acal'),'Q04':('joint','M'),'Q07':('joint','RAFT'),'Q08':('joint','Chronos')}
def selected_receipt(protocol,seed):
    params=json.loads((BASE/'learning/hyperparameters.json').read_bytes())[protocol]
    run=f"{protocol}_s{seed}_l{params['lambda_risk']:g}_q{params['eta_queue']:g}"
    return BASE/'learning/formal'/run/'summary.json'
@lru_cache
def policy(protocol,seed):
    from aoi_policy import Policy
    record=json.loads(selected_receipt(protocol,seed).read_bytes())
    if record['pilot'] or record['policy_seed']!=seed or record['protocol']!=protocol or record['gradient_steps']!=30000 or record['validation_episodes']!=96:
        raise ValueError('Formal training and all three validation checkpoints are required')
    path=Path(record['model'])
    net=Policy().eval();net.load_state_dict(torch.load(path,weights_only=True))
    return net,sha_file(path)

def stream(w,world_hash,protocol,seed,b,q):
    from aoi_reference import QuotaGrid
    net,psha=policy(protocol,seed)
    key=f'{world_hash}_{protocol}_{seed}_B{b:g}_K{q}_{psha[:12]}'
    dest=OUT/'streams'/(key+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(dest)+'.lock'):
        if not dest.exists():
            grid=QuotaGrid(budget_us=round(b*1e6),quota=q)
            es,_=plan_events(w,grid,protocol,net)
            sha=hashlib.sha256(json.dumps(es,separators=(',',':')).encode()).hexdigest()
            put(dest,dict(world=world_hash,seed=seed,policy_sha256=psha,events=es,event_sha256=sha))
        r=json.loads(dest.read_bytes())
        if (r['world'],r['policy_sha256'],len(r['events']))!=(world_hash,psha,q):raise ValueError(key)
        if hashlib.sha256(json.dumps(r['events'],separators=(',',':')).encode()).hexdigest()!=r['event_sha256']:raise ValueError('stored event digest')
    return r

def series(w,world_hash,split,k,row,seed,b,q,scales,folder):
    import paid_full as pf
    from closed import simulate_controls
    protocol,kind=ROWS[row];frozen=stream(w,world_hash,protocol,seed,b,q)
    es=frozen['events']
    todo=[]
    for m in scales:
        key=f'{row}_{k}_s{seed}_B{b:g}_K{q}_m{m:.2f}';path=folder/'receipts'/(key+'.json')
        if path.exists():
            r=json.loads(path.read_bytes())
            if (r['world'],r['event_sha256'],r['policy_sha256'])!=(world_hash,frozen['event_sha256'],frozen['policy_sha256']):raise ValueError(key)
        else:todo.append((m,key,path))
    if not todo:return
    model=Model(w,kind);predict=[]
    for u,start,end in es:
        obs=w.observe(u,start/1e6+.0031)
        predict.append((u,obs.t,*model.predict(u,obs)))
    for m,key,path in todo:
        idx=0
        def cached(u,o):
            nonlocal idx
            pred=predict[idx];idx+=1
            if pred[0]!=u or pred[1]!=o.t:raise ValueError('forecast issued at different paid observation')
            return pred[2],pred[3]
        mark=folder/'active'/path.name;put(mark,dict(pid=os.getpid(),started=time.time()))
        try:
            observed,r,_=simulate_controls(w,'stream',cached,model.factors,pf,b,m,q,event_stream=es)
            if idx!=q or len(observed)!=q or abs(r['ledger_s'][0]-q*.0031)>1e-7:raise ValueError('matched cost')
            put(path,dict(key=key,split=split,protocol='g1_linux_paired_v2',world=world_hash,world_id=k,row=row,
                seed=seed,budget_s=b,quota=q,scale=m,predictor=kind,event_sha256=frozen['event_sha256'],
                policy_sha256=frozen['policy_sha256'],events=observed,result=r))
        finally:mark.unlink(missing_ok=True)
def work(job):
    k,row,seed,b,q=job
    with compute_slot():
        w,meta=load_checked('development',k);w.cfg['simulation']['sensing_budget']=b/180
        series(w,meta['physical_sha256'],'development',k,row,seed,b,q,SCALES,OUT)
def main():
    from collections import Counter
    initialize()
    from closed import freeze
    for name in ('receipts','active','streams'):(OUT/name).mkdir(parents=True,exist_ok=True)
    todo=[(k,row,seed,b,q) for i,(b,q) in enumerate(budgets()) for k in range(1400,1408)
          for row in (list(ROWS) if i==0 else ['Q03']) for seed in (17,29,41)]
    todo=list(dict.fromkeys(todo));start=time.time()
    put(OUT/'batch.json',dict(total=len(todo)*len(SCALES),workers=40,pid=os.getpid(),start=start,phase='development'))
    ctx=multiprocessing.get_context('fork')
    pending=todo.copy();running=set()
    with ProcessPoolExecutor(16,mp_context=ctx) as heavy,ProcessPoolExecutor(24,mp_context=ctx) as other:
        while pending or running:
            released={key for key in {(ROWS[j[1]][0],j[2]) for j in pending} if selected_receipt(*key).is_file()}
            ready=[j for j in pending if (ROWS[j[1]][0],j[2]) in released]
            for j in ready:
                running.add((heavy if j[1]=='Q08' else other).submit(work,j))
                pending.remove(j)
            if running:
                finished,running=wait(running,timeout=10,return_when=FIRST_COMPLETED)
                for future in finished:future.result()
            elif pending:
                time.sleep(10)
    groups={}
    for k,row,seed,b,q in todo:
        for m in SCALES:
            r=json.loads((OUT/'receipts'/f'{row}_{k}_s{seed}_B{b:g}_K{q}_m{m:.2f}.json').read_bytes())
            groups.setdefault((row,b,q),{}).setdefault(m,[]).append(r)
    policies={f'{row}_B{b:g}_K{q}':freeze(by) for (row,b,q),by in groups.items()}
    put(OUT/'summary.json',dict(total=len(todo)*4,policies=policies,elapsed_s=time.time()-start))
    put(OUT/'batch.json',dict(total=len(todo)*4,workers=40,pid=os.getpid(),start=start,phase='done'))
if __name__=='__main__':main()
