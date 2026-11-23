"""Resumable E22-LGB execution for the approved matched-center comparison."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import BASE,ROOT,ASSETS
from .network import execute

MODELS=('chronos','moirai','vector')
PY=sys.executable
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(module,*args):
    name='_'.join((module,*map(str,args))).replace('--','')
    with (BASE/'logs'/(name+'.log')).open('a') as f:
        subprocess.run([PY,'-m','refine.'+module,*map(str,args)],cwd=ROOT,check=True,stdout=f,stderr=subprocess.STDOUT)

def state(stage):
    p=BASE/'tube_campaign_status.json'
    temp=p.with_suffix('.part')
    temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S %z'))))
    temp.replace(p)
    print(stage,flush=True)

def build(name):
    if name=='vector' and not (BASE/'weights/vector7.pt').exists():run('fit_vector')
    if not (BASE/'weights'/(name+'_fit.json')).exists():run('tubes',name)

def check_model(name):
    run('offline',name)
    run('network',name,'--phase','validation','--count',1)
    pairs=[json.loads((BASE/'tube_network/validation'/name/'000'/(t+'.json')).read_text()) for t in ('native','lightgbm')]
    for row in pairs:
        assert row['probes_per_user']==9000 and len(row['bits'])==16
        assert 0<=row['R']<=1 and row['G_mbps']>=0
    assert pairs[0]['initial_center_sha256']==pairs[1]['initial_center_sha256']

def freeze():
    files=list((ROOT/'refine').glob('*.py'))+list((ROOT/'routeclock').glob('*.py'))+list((ROOT/'configs').glob('*.json'))
    files+=list((BASE/'weights').glob('*'))+[ASSETS/'SHA256SUMS',BASE/'planned_cells.csv']
    signature={str(f):digest(f) for f in sorted(files) if f.is_file()}
    data_hash=hashlib.sha256()
    for f in sorted((BASE/'observations').glob('*/*.npz')):
        data_hash.update(f.relative_to(BASE).as_posix().encode());data_hash.update(digest(f).encode())
    config=dict(files=signature,observations_sha256=data_hash.hexdigest(),world_namespace='refine_v1/confirmation',
                ids=list(range(70)),models=MODELS,tubes=['native','lightgbm','fixed'],seed=7,
                conditioning='shared external world; independent observation histories after initial service',
                endpoints=['R_lightgbm - R_native','G_lightgbm - G_native'],bootstrap_seed=271828,
                additional_lightgbm_count=210,native_and_fixed_existing_budget=420,
                validation='one full paired commissioning world per model; no selection using confirmation')
    file=BASE/'e22_freeze.json'
    if file.exists():
        if json.loads(file.read_text())!=json.loads(json.dumps(config)):raise RuntimeError('frozen inputs changed')
    else:
        temp=file.with_suffix('.part');temp.write_text(json.dumps(config,indent=2));temp.replace(file)

def results():
    rows=[json.loads(f.read_text()) for f in sorted((BASE/'tube_network/confirmation').glob('*/*/*.json'))]
    frame=pd.DataFrame([{k:r[k] for k in ['model','tube','world','R','G_mbps','unavailable_fraction','narrow_good_data_s','initial_center_sha256','wall_s']} for r in rows])
    out=BASE/'e22_results';out.mkdir(exist_ok=True)
    frame.to_csv(out/'worlds.csv',index=False)
    summary=[]
    for name in MODELS:
        subset=frame[frame.model==name]
        assert len(subset)==210 and subset.groupby('world').initial_center_sha256.nunique().max()==1
        for field in ['R','G_mbps','narrow_good_data_s','unavailable_fraction']:
            wide=subset.pivot(index='world',columns='tube',values=field)
            for arm in ['lightgbm','fixed']:
                delta=(wide[arm]-wide.native).to_numpy();assert len(delta)==70 and np.isfinite(delta).all()
                rng=np.random.default_rng(271828);draw=rng.choice(delta,(5000,len(delta))).mean(1)
                low,high=np.quantile(draw,[.025,.975])
                summary.append(dict(model=name,contrast=arm+' - native',metric=field,worlds=len(delta),mean=delta.mean(),lo95=low,hi95=high))
    pd.DataFrame(summary).to_csv(out/'paired_summary.csv',index=False)

def worker(name,first,last):
    execute(name,'confirmation',range(first,last),('native','lightgbm','fixed'),BASE/'tube_network')

def main():
    lock=(BASE/'e22_campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state('prepare_observations')
    # The initial reconstruction may already be running when the queue is started.
    while len(list((BASE/'observations').glob('*/*.npz')))<2000:
        procs=[p for p in Path('/proc').glob('[0-9]*/cmdline')]
        running=False
        for p in procs:
            try: args=p.read_bytes().split(b'\0')
            except OSError:continue
            if b'refine.data' in args:running=True;break
        if not running:run('data')
        else:time.sleep(15)
    state('fit_centers_and_tubes')
    with concurrent.futures.ThreadPoolExecutor(3) as ex:list(ex.map(build,MODELS))
    state('full_validation')
    with concurrent.futures.ThreadPoolExecutor(3) as ex:list(ex.map(check_model,MODELS))
    state('freeze_and_confirmation')
    freeze()
    # Each process retains one checkpoint. Distinct shards write disjoint world IDs.
    jobs=[(name,0,35) for name in MODELS]+[(name,35,70) for name in MODELS]
    def launch(job):
        name,lo,hi=job
        with (BASE/'logs'/f'confirm_{name}_{lo}.log').open('a') as f:
            subprocess.run([PY,'-m','refine.campaign','worker',name,str(lo),str(hi)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(6) as ex:list(ex.map(launch,jobs))
    results();state('E22-LGB_complete')
if __name__=='__main__':
    if len(sys.argv)>1:
        worker(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
    else:
        try:main()
        except Exception as exc:
            state('error: '+repr(exc))
            raise
