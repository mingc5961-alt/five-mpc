"""Dispatch the remaining plan cells, independent search grid and prerequisite checks."""
import argparse
from contextlib import ExitStack
import csv
import fcntl
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from refine.data import BASE,ROOT
from runtime.learned_queue import digest,save
from runtime.remaining_models import OUT,ARCHIVED,classic_params,gate_params,raw_params
from runtime.remaining_runs import Runner,fixed_choice

NAMES=('F','I','P','RC','CR-G','CR_raw')
def plan():
    with (BASE/'planned_cells.csv').open() as f:rows=list(csv.DictReader(f))
    formal=[]
    for n,r in enumerate(rows,1):
        if r['model'] not in NAMES:continue
        formal.append(dict(r,id=f'RC-{n:04}',world=int(r['world']),seed=int(r['seed']),mcs_margin_db=int(r['mcs_margin_db'])))
    assert len(formal)==len({r['id'] for r in formal})==1250
    search=[dict(id=f'grid-{w}-{s}-{h}',world=w,model='RC',tube='default',phase='search',seed=7,
                 condition='T',mcs_margin_db=5,risk_scale=s,max_interval=h)
            for w in range(10) for s in (.85,.925,1.,1.15,1.30) for h in (2,3,5,8,10)]
    checks=[dict(id='check-'+name,world=0,model=name,tube=tube,phase=phase,seed=7,condition='T',mcs_margin_db=5)
            for name,tube,phase in [('F','default','search'),('RC','fixed','validation'),('CR_raw','default','validation')]]
    return formal,search,checks

def artifact(name):
    return OUT/({'F':'F.json','CR-G':'CR-G.pkl','CR_raw':'CR_raw.pkl'}.get(name,'classical.pkl'))

def prepare(name):
    with (OUT/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=OUT/(name+('.json' if name=='F' else '.pkl'))
        if not path.exists():
            if name=='F':save(path,fixed_choice())
            else:{'CR-G':gate_params,'CR_raw':raw_params,'classical':classic_params}[name]()
        meta=dict(artifact=digest(path),name=name,archived_sha256=digest(ARCHIVED))
        if name!='F':
            data=pickle.loads(path.read_bytes())
            bundles=list(data.values()) if name=='classical' else [data]
            import numpy as np
            assert all(np.asarray(f).shape==(7,) and np.isfinite(f).all() for b in bundles for f in b['factors'].values())
        save(OUT/(name+'_ready.json'),meta)

def product(r):
    return OUT/('results' if r['id'].startswith('RC-') else 'ancillary')/(r['id']+'.json')

def locked(stack,path):
    lock=stack.enter_context(path.open('a'))
    try:fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:return False
    return True

def process_count(module):
    n=0
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().split(b'\0')
        except OSError:continue
        if len(args)>=4 and args[1:3]==[b'-m',module] and args[3] in (b'worker',b'prepare'):n+=1
    return n

def record(row,runner,formal):
    import torch
    marker=OUT/'active'/f'{os.getpid()}.json'
    started=time.time();save(marker,dict(row,pid=os.getpid(),started=started))
    try:
        cpu=time.process_time();tic=time.monotonic()
        gpu=row['model'] in ('CR_raw','CR-G')
        if gpu:torch.cuda.reset_peak_memory_stats()
        data=runner.execute(row)
        if row['id']=='check-F':
            import numpy as np
            s=data['fixed']
            with (BASE/'q1_frontier/F_scores.csv').open() as stream:
                ref=next(r for r in csv.DictReader(stream) if int(r['world'])==0 and
                         [int(r['width_x']),int(r['width_y'])]==s['pair'] and float(r['period_s'])==s['period'])
            for k in ('R','G_mbps','narrow_good_data_s'):
                np.testing.assert_allclose(data[k],float(ref[k]),rtol=1e-9,atol=1e-9)
        if row['id']=='check-RC' or (row['model']=='RC' and row['tube']=='fixed'):
            ref=next(r for r in formal if r['model']=='RC' and r['phase']==row['phase'] and r['world']==row['world']
                     and r['condition']=='T' and r['tube']=='default' and r['mcs_margin_db']==5)
            assert data['initial_center_sha256']==json.loads(product(ref).read_text())['initial_center_sha256']
        data.update(wall_s=time.monotonic()-tic,cpu_seconds=time.process_time()-cpu,pid=os.getpid(),started=started,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if gpu else 0.,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if gpu else 0.)
        save(product(row),data)
        print(row['id'],round(data['wall_s'],2),flush=True)
    finally:marker.unlink(missing_ok=True)

def worker(slot):
    formal,search,checks=plan(); runner=Runner()
    validations={m:{r['id'] for r in formal if r['phase']=='validation' and r['model']==m} for m in NAMES}
    counterpart={(r['phase'],r['world']):r['id'] for r in formal if r['model']=='RC' and r['tube']=='default' and r['condition']=='T' and r['mcs_margin_db']==5}
    seq=checks+formal+search
    while True:
        done={p.stem for folder in ['results','ancillary'] for p in (OUT/folder).glob('*.json')}
        if len(done)==len(seq):return
        # Share nine CPU cores with the other queue, expand after its GPU workers drain.
        if slot>=max(2,8-process_count(b'runtime.learned_queue')):
            time.sleep(4);continue
        ready={m for m in NAMES if (OUT/(artifact(m).stem+'_ready.json')).exists()}
        eligible=[]
        for row in seq:
            n=row['id'];m=row['model']
            if n in done or m not in ready:continue
            if m=='F' and n!='check-F' and 'check-F' not in done:continue
            if m=='CR_raw' and n!='check-CR_raw' and 'check-CR_raw' not in done:continue
            if row['tube']=='fixed' and (counterpart[(row['phase'],row['world'])] not in done or
                   (n!='check-RC' and 'check-RC' not in done)):continue
            if row['phase']=='confirmation' and not validations[m].issubset(done):continue
            eligible.append(row)
        found=False
        # Validate first. Within each stage rotate by worker to avoid queue contention.
        groups=[[r for r in eligible if r['id'].startswith('check-')],
                [r for r in eligible if r['phase']=='validation'],
                [r for r in eligible if r['phase']=='confirmation'],
                [r for r in eligible if r['id'].startswith('grid-')]]
        for group in groups:
            for row in group[slot:]+group[:slot]:
                with ExitStack() as stack:
                    if not locked(stack,OUT/'locks'/(row['id']+'.lock')) or product(row).exists():continue
                    if row['model'] in ('CR-G','CR_raw') and not any(locked(stack,OUT/'locks'/f'gpu_{i}.lock') for i in range(2)):continue
                    record(row,runner,formal);found=True;break
            if found:break
        if not found:time.sleep(2)

def run(*args):
    log=BASE/'logs'/('remaining_'+'_'.join(map(str,args))+'.log')
    with log.open('a') as f:
        for attempt in range(2):
            cp=subprocess.run([sys.executable,'-m','runtime.full_queue',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            if cp.returncode==0:return
            print('failed attempt',attempt+1,cp.returncode,file=f,flush=True)
    raise RuntimeError(str(log))

def main():
    from refine import campaign
    for sub in ('','active','results','ancillary','locks'): (OUT/sub).mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign.freeze()
        formal,search,checks=plan()
        with (BASE/'planned_cells.csv').open() as f:plan_rows=list(csv.DictReader(f))
        old=set()
        for i,r in enumerate(plan_rows,1):
            if r['model'] not in ('chronos','vector','moirai') or r['seed']!='7' or r['condition']!='T':continue
            if r['phase']=='confirmation' or (r['world']=='0' and r['phase']=='validation' and r['tube']=='default'):
                tube='native' if r['tube']=='default' else r['tube']
                path=BASE/'tube_network'/r['phase']/r['model']/f"{int(r['world']):03}"/(tube+'.json')
                assert path.exists()
                old.add(f'RC-{i:04}')
        learned={r['id'] for r in json.loads((BASE/'learned_queue/manifest.json').read_text())['cells']}
        ours={r['id'] for r in formal}
        assert len(old)==633 and len(old|learned|ours)==2280 and len(old)+len(learned)+len(ours)==2280
        hashes={p.name:digest(p) for p in Path(__file__).parent.glob('remaining_*.py')}
        hashes['full_queue.py']=digest(Path(__file__))
        save(OUT/'manifest.json',dict(pid=os.getpid(),cells=formal,search=search,checks=checks,hashes=hashes,
              previous=len(old),learned=len(learned),total=2280,started=time.time(),plan_sha256=digest(BASE/'planned_cells.csv')))
        with ThreadPoolExecutor(12) as pool:
            futures=[pool.submit(run,'prepare',m) for m in ['F','classical','CR-G','CR_raw']]
            futures.extend(pool.submit(run,'worker',i) for i in range(8))
            from concurrent.futures import as_completed
            for f in as_completed(futures):
                try:f.result()
                except Exception as exc:
                    save(OUT/'error.json',dict(time=time.time(),error=str(exc)))
                    raise
        assert all(product(r).is_file() for r in formal+search+checks)
        import pandas as pd
        for rows,path in [(formal,OUT/'results.csv'),(search,BASE/'q1_frontier/RC_scores.csv')]:
            frame=pd.DataFrame([{k:v for k,v in json.loads(product(r).read_text()).items() if isinstance(v,(int,float,str))} for r in rows])
            tmp=path.with_suffix('.part');frame.to_csv(tmp,index=False);tmp.replace(path)
        save(OUT/'complete.json',dict(cells=len(formal),search=len(search),time=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','worker','prepare'));p.add_argument('args',nargs='*');a=p.parse_args()
    if a.action=='run':main()
    elif a.action=='prepare':prepare(a.args[0])
    else:worker(int(a.args[0]))
