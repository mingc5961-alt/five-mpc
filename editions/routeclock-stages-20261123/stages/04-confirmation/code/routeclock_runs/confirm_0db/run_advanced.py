import concurrent.futures
import subprocess
import sys
import fcntl
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from advanced_tubes import Predictor,get_bundle,ROOT,EPS
from refine import network,world_cache
from runtime.remaining_runs import controller,BUDGET
from routeclock.config import load_config
from routeclock.radio import Radio

def cells(net,seed,bundle,stage,methods,world_ids=None):
    scope=json.loads((ROOT/'scope.json').read_text())
    count=scope[stage]['worlds']
    namespace=scope['analysis'][stage+'_namespace']
    world_cache.BASE=ROOT/'work/baseline_cache'
    # New LightGBM head already predicts a quantile in radians.
    def tree_scale(tree,features):
        return np.maximum(EPS,tree.predict(features.reshape(-1,features.shape[-1]),num_threads=1).reshape(-1,21,2))
    network.tree_scale=tree_scale
    for world_id in (range(count) if world_ids is None else world_ids):
        config=load_config();config['simulation']['sensing_budget']=BUDGET['T']
        world=world_cache.paired_world(config,namespace,world_id)
        for budget,rho in BUDGET.items():
            cfg=load_config();cfg['simulation']['sensing_budget']=rho
            radio=Radio(cfg);radio.threshold-=5
            hashes=set()
            for method in methods:
                folder=ROOT/'results'/stage/net.name/str(seed)/f'{world_id:03}_{budget}_{method}'
                folder.mkdir(parents=True,exist_ok=True)
                done=folder/'complete.json'
                if done.exists():
                    old=json.loads(done.read_text())
                    assert old['center_sha256']==net.sha and old['namespace']==namespace
                    hashes.add(old['result']['initial_center_sha256'])
                    continue
                start=time.monotonic()
                with controller(radio,10.):
                    result,_=network.run_world(cfg,world,net,bundle,method)
                assert abs(sum(result['ledger_s'])-180)<1e-6 and result['probes_per_user']==9000
                hashes.add(result['initial_center_sha256'])
                record=dict(model=net.name,seed=seed,center_sha256=net.sha,namespace=namespace,world=world_id,budget=budget,tube=method,mcs_margin_db=0,risk_scale=1,max_interval_s=10,result=result,elapsed_s=time.monotonic()-start)
                tmp=folder/'complete.tmp'
                tmp.write_text(json.dumps(record,allow_nan=False));tmp.replace(done)
            assert len(hashes)==1

def run(name):
    net=Predictor(name,7);bundle=get_bundle(net,7)
    candidates=bundle['candidates']
    cells(net,7,bundle,'development',candidates)
    scores={m:[] for m in candidates}
    for p in (ROOT/'results/development'/name/'7').glob('*/complete.json'):
        row=json.loads(p.read_text())
        scores[row['tube']].append(row['result']['G_mbps'])
    assert all(len(v)==30 for v in scores.values())
    winner=min(scores,key=lambda m:(-np.mean(scores[m]),m))
    (ROOT/'tubes'/name/'selection.json').write_text(json.dumps(dict(tube=winner,development_mean_mbps={m:np.mean(v) for m,v in scores.items()})))
    cells(net,7,bundle,'confirmation',[winner])
    del net
    for seed in (11,19):
        net=Predictor(name,seed);bundle=get_bundle(net,seed)
        cells(net,seed,bundle,'confirmation',[winner])
        del net
if __name__=='__main__':
    if len(sys.argv) == 2:
        run(sys.argv[1])
        sys.exit(0)
    with (ROOT/'state/advanced.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Three independent workers share one GPU and leave six CPU slots for baselines.
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            jobs={pool.submit(subprocess.run,[sys.executable,str(Path(__file__).resolve()),name],check=True):name for name in ('VECTOR','RAFT','Chronos-2')}
            for job in concurrent.futures.as_completed(jobs):
                job.result()
