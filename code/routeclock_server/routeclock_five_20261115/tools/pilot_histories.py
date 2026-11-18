"""Recreate the previous pilot's physical worlds and retain causal long-history snapshots."""
import argparse
import hashlib
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from routeclock.config import load_config
from routeclock.predictors import Tracker
from routeclock.world import World,GROUPS
from routeclock.foundation import ChronosAdapter
from routeclock.geometry import unit

OUT=ROOT/'runs/five_pilot'
def configuration():
    cfg=load_config();cfg['simulation'].update(primary_N=3,warmup_s=180,service_s=18)
    return cfg

from routeclock.event_labels import event_at

def prepare(job):
    split,index=job
    path=OUT/f'{split}_{index:02}.pkl'
    if path.exists():return path.name
    cfg=configuration();network=split=='network'
    source='train' if split=='validation' else split
    groups=[GROUPS[(index*3+i)%6] for i in range(3)] if network else [GROUPS[index%6]]
    world=World(cfg,f'long-history-deadline-v1/{source}',index,n=len(groups),groups=groups)
    records=[];trackers=[]
    for uid,g in enumerate(groups):
        tracker=Tracker(cfg)
        times=np.arange(180)+.5+.1*uid
        for t in times:
            tracker.observe(world.observe(uid,t))
            pkg=tracker.issue()
        features=tracker.features(pkg,'P')
        inputs=[ChronosAdapter.inputs(tracker.history,tracker.snapshots,pkg.centers['P'],
                np.sqrt(np.maximum(0,np.diagonal(pkg.covariance['P'],axis1=1,axis2=2))),pkg.diagnostics['support'],flag) for flag in (False,True)]
        hs=np.arange(501)*.02;truth=np.array([world.truth(uid,t+h)[0] for h in hs])
        event,distance=event_at(world,uid,t)
        records.append(dict(task=f'{source}/{index}/{uid}',group=g,package=pkg,features={k:features for k in ['I','P','C','CR_raw']},
                       truth_h=hs,truth=truth,inputs=inputs,event=event,route_distance_m=distance,
                       route_hash=hashlib.sha256(world.references[uid].tobytes()).hexdigest()))
        if network:trackers.append(tracker)
    target=path.with_suffix('.tmp')
    with target.open('wb') as stream:pickle.dump((records,trackers,world if network else None),stream)
    target.replace(path);return path.name

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(s,i) for s,ids in [('train',range(12)),('validation',range(12,18)),('cal',range(12)),('network',range(2))] for i in ids]
    with ProcessPoolExecutor(args.workers) as pool:
        for name in pool.map(prepare,jobs):print(name,flush=True)

