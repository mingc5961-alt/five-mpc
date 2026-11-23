"""Check archived initial predictions and replay cells with a different current-device fingerprint."""
import hashlib
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

out=Path(__file__).resolve().parent
root=Path('/data/chen/routeclock_runs/confirm_0db')
sys.path.insert(0,str(root))


def one(model):
    from advanced_tubes import Predictor,get_bundle,EPS
    from refine import network,world_cache
    from routeclock.config import load_config
    from routeclock.radio import Radio
    from runtime.remaining_runs import BUDGET,controller
    from parallel_frontier import frontier
    from run_paired import namespace
    world_cache.BASE=root/'work/baseline_cache'
    network.frontier=frontier
    def quantile(tree,x):
        return np.maximum(EPS,tree.predict(x.reshape(-1,x.shape[-1]),num_threads=1).reshape(-1,21,2))
    network.tree_scale=quantile
    model_obj=Predictor(model,7)
    bundle=get_bundle(model_obj,7)
    folder=out/'先进模型硬件复核'/model
    folder.mkdir(parents=True,exist_ok=True)
    checks=[]
    for w in range(10):
        cfg=load_config()
        cfg['simulation']['sensing_budget']=BUDGET['T']
        world=world_cache.paired_world(cfg,namespace,w)
        initial=model_obj.batch(network.warm_history(world,180.,.0031))
        fingerprint=hashlib.sha256(np.stack([x for x,_ in initial]).tobytes()).hexdigest()
        for p in sorted((root/'results/development'/model/'7').glob(f'{w:03}_*/complete.json')):
            rec=json.loads(p.read_text())
            assert rec['namespace']==namespace and rec['center_sha256']==model_obj.sha
            match=rec['result']['initial_center_sha256']==fingerprint
            checks.append([w,rec['budget'],rec['tube'],match])
            if match:
                continue
            path=folder/(p.parent.name+'.json')
            if path.exists():
                continue
            b,t=rec['budget'],rec['tube']
            cfg=load_config();cfg['simulation']['sensing_budget']=BUDGET[b]
            radio=Radio(cfg);radio.threshold-=5
            start=time.monotonic()
            with controller(radio,10.):
                val,_=network.run_world(cfg,world,model_obj,bundle,t)
            assert val['initial_center_sha256']==fingerprint
            new=dict(rec,result=val,elapsed_s=time.monotonic()-start,original_receipt=str(p))
            tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(new,allow_nan=False));tmp.replace(path)
        print(model,w,sum(not row[-1] for row in checks),flush=True)
    (folder/'预热复核.json').write_text(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(one,['Chronos-2','VECTOR','RAFT']))
