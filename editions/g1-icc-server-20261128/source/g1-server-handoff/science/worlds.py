"""Generate immutable world inputs for the Linux reconstruction namespace."""
import argparse
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from server_paths import RUN
from routeclock.config import load_config
from routeclock.world import GROUPS, World

COUNTS = (3, 1, 1, 1, 3, 7)
SPLITS = {'train':range(1000,1064), 'train_validation':range(1100,1116),
          'tube_fit':range(1200,1216), 'tube_calibration':range(1300,1316),
          'development':range(1400,1408), 'confirmation':range(2000,2024), 'historical':range(1),
          'rebuild_holdout':range(4000,4024)}


def fingerprint(w):
    h = hashlib.sha256()
    for array in (w.positions, np.asarray(w.noise), np.asarray(w.shadows)):
        h.update(array.tobytes())
    return h.hexdigest()


def generate(job):
    split, k = job
    target = RUN/'worlds'/f'{split}_{k}.pkl'
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = target.with_suffix('.json')
    if target.exists() and summary.exists():
        return
    ns = 'signed_g1_factorial_v1/confirm' if split == 'historical' else 'g1_icc_server_v1/main/'+split
    if split == 'rebuild_holdout':
        ns='g1_linux_refit_v1/main/holdout'
    cfg = load_config()
    cfg['simulation'].update(warmup_s=0,service_s=180,sensing_budget=3/180)
    groups = np.repeat(GROUPS, COUNTS)[np.random.default_rng(410827+k).permutation(16)].tolist()
    w = World(cfg, ns, k, groups=groups)
    file = target.with_suffix(f'.{os.getpid()}.tmp')
    with file.open('wb') as f:
        pickle.dump(w,f,protocol=5)
    file.replace(target)
    temp = summary.with_suffix(f'.{os.getpid()}.json.tmp')
    temp.write_text(json.dumps(dict(split=split,id=k,namespace=ns,physical_sha256=fingerprint(w),groups=groups)))
    temp.replace(summary)


def load_world(split,k):
    path = RUN/'worlds'/f'{split}_{k}.pkl'
    with path.open('rb') as f:
        return pickle.load(f)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('splits',nargs='+',choices=list(SPLITS))
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args()
    jobs=[(s,k) for s in a.splits for k in SPLITS[s]]
    with ProcessPoolExecutor(a.workers) as pool:
        for i,_ in enumerate(pool.map(generate,jobs),1):
            if i%8==0 or i==len(jobs):print(i,len(jobs),flush=True)
