"""Resume learned-model plan cells using the frozen E22 controller and isolated seed weights."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from refine.data import BASE, ROOT

DEST=BASE/'learned_queue'
MODELS=('chronos','moirai','vector')
def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,allow_nan=False))
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def jobs():
    with (BASE/'planned_cells.csv').open() as f:
        for i,row in enumerate(csv.DictReader(f),1):
            phase,condition,model,seed=row['phase'],row['condition'],row['model'],int(row['seed'])
            world=int(row['world'])
            if model not in MODELS: continue
            if phase=='validation' and seed==7 and condition in ('T','L'):
                if condition=='T' and world==0: continue
            elif not (phase=='confirmation' and model=='vector' and seed in (11,19) and condition=='T'):
                continue
            yield dict(id=f'RC-{i:04}',phase=phase,condition=condition,model=model,seed=seed,world=world,
                       mcs_margin_db=int(row['mcs_margin_db']),tube={'default':'native','fixed':'fixed'}[row['tube']])

def prepare(seed):
    from refine import fit_vector, models, tubes
    import numpy as np
    root=DEST/f'seed{seed}'
    weights=root/'weights'
    weights.mkdir(parents=True,exist_ok=True)
    ck=weights/f'vector{seed}.pt'
    proof=root/'ready.json'
    if proof.exists():
        meta=json.loads(proof.read_text())
        assert meta['checkpoint']==digest(ck) and meta['bundle']==digest(weights/'vector_tubes.pkl')
        return
    if not (ck.is_file() and (weights/f'vector{seed}_fit.json').is_file()):
        fit_vector.BASE=root
        fit_vector.main(seed)
    model=models.Vector(ck)
    train=[p for p in tubes.paths('train') if int(p.stem)%5==0]
    cal=tubes.paths('calibration')
    val=tubes.paths('validation')
    assert [len(train),len(cal),len(val)]==[240,400,400]
    et=tubes.slim_predictions(model,train)[3]
    _,_,_,ec,native,_=tubes.slim_predictions(model,cal)
    _,_,_,ev,vnative,_=tubes.slim_predictions(model,val)
    fixed=np.maximum(tubes.EPS,np.median(et[:,::25],axis=0))
    bundle=dict(model='vector',fixed=fixed,factors={})
    coverage={}
    for tube,raw,vraw in [('native',native,vnative),('fixed',np.broadcast_to(fixed,native.shape),np.broadcast_to(fixed,vnative.shape))]:
        scores=tubes.scores(ec,raw)
        factor=np.maximum.accumulate([tubes.finite_quantile(v,.1) for v in scores.T])
        assert np.isfinite(factor).all()
        bundle['factors'][tube]=factor
        coverage[tube]=np.mean(tubes.scores(ev,vraw)<=factor,axis=0).tolist()
    target=weights/'vector_tubes.pkl'
    temp=target.with_suffix('.part')
    temp.write_bytes(pickle.dumps(bundle))
    temp.replace(target)
    save(proof,dict(seed=seed,checkpoint=digest(ck),bundle=digest(target),coverage=coverage,calibration_routes=len(cal)))

def worker(model_name,seed,shard):
    import torch
    from refine import network
    from routeclock.config import load_config
    from runtime.fast_ledger import DeferredLedger
    weights=(BASE if seed==7 else DEST/f'seed{seed}')/'weights'
    ck=weights/f'vector{seed}.pt'
    model=network.MODELS[model_name](ck) if model_name=='vector' else network.MODELS[model_name]()
    bundle_path=weights/(model_name+'_tubes.pkl')
    bundle=pickle.loads(bundle_path.read_bytes())
    bundle_sha=digest(bundle_path)
    network.Ledger=DeferredLedger
    rows=[r for r in jobs() if (r['model'],r['seed'],r['world']%2)==(model_name,seed,shard)]
    marker=DEST/'active'/f'{os.getpid()}.json'
    for row in rows:
        path=DEST/'results'/(row['id']+'.json')
        if path.exists():
            data=json.loads(path.read_text())
            assert all(data[k]==v for k,v in row.items()) and data['bundle_sha256']==bundle_sha
            continue
        cfg=load_config()
        cfg['simulation']['sensing_budget']=.05/3
        world=network.paired_world(cfg,'refine_v1/'+row['phase'],row['world'])
        # Sensing permission only changes the controller. Keep the paired physical world and warm history.
        cfg['simulation']['sensing_budget']={'T':.05/3,'L':.05}[row['condition']]
        started=time.time()
        save(marker,dict(**row,pid=os.getpid(),started=started))
        try:
            torch.cuda.reset_peak_memory_stats()
            tic=time.monotonic(); cpu=time.process_time()
            result,_=network.run_world(cfg,world,model,bundle,row['tube'])
            result.update(row,config=cfg['simulation'],bundle_sha256=bundle_sha,wall_s=time.monotonic()-tic,
                          started=started,pid=os.getpid(),cpu_seconds=time.process_time()-cpu,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            assert result['probes_per_user']==9000 and len(result['bits'])==16
            if row['tube']=='fixed':
                sibling=next(r for r in rows if r['world']==row['world'] and r['tube']=='native')
                ref=json.loads((DEST/'results'/(sibling['id']+'.json')).read_text())
                assert result['initial_center_sha256']==ref['initial_center_sha256']
            save(path,result)
            print(row['id'],round(result['wall_s'],2),flush=True)
        finally:
            marker.unlink(missing_ok=True)

def run(*args):
    with (BASE/'logs'/('learned_'+'_'.join(map(str,args))+'.log')).open('a') as log:
        cmd=[sys.executable,'-m','runtime.learned_queue',*map(str,args)]
        for attempt in range(2):
            if subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode==0:return
    raise RuntimeError(f'failed {args}')

def main():
    from refine import campaign
    DEST.mkdir(exist_ok=True)
    with (DEST/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign.freeze()
        rows=list(jobs())
        assert len(rows)==len({r['id'] for r in rows})==397
        save(DEST/'manifest.json',dict(pid=os.getpid(),cells=rows,freeze=digest(BASE/'e22_freeze.json'),runner=digest(Path(__file__))))
        def seed_chain(seed):
            run('prepare',seed)
            # Both tubes share the same world/shard.
            with ThreadPoolExecutor(2) as ex:
                list(ex.map(lambda shard:run('worker','vector',seed,shard),range(2)))
        with ThreadPoolExecutor(6) as ex:
            seeds=[ex.submit(seed_chain,s) for s in (11,19)]
            workers=[ex.submit(run,'worker',m,7,h) for h in range(2) for m in MODELS]
            for future in workers+seeds: future.result()
        assert all((DEST/'results'/(r['id']+'.json')).is_file() for r in rows)
        save(DEST/'complete.json',dict(cells=len(rows),time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('run','prepare','worker'));p.add_argument('args',nargs='*');a=p.parse_args()
    if a.action=='run':main()
    elif a.action=='prepare':prepare(int(a.args[0]))
    else:worker(a.args[0],int(a.args[1]),int(a.args[2]))
