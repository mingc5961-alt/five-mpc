"""Build the immutable Linux world bank shared by all locally paired branches."""
import concurrent.futures as futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
from history import ROOT,PACKAGE,put

BANK=ROOT/'paired_linux_v1/worlds'
CPU=PACKAGE/'gru4505'
ENGINE=CPU/'confirm_0db/work/code/routeclock_refine'
sys.path.insert(0,str(ENGINE))
import numpy as np
from routeclock.world import GROUPS,World
from routeclock.config import load_config

SPLITS={'development':range(1400,1408),'train':range(1000,1064),'train_validation':range(1100,1116),
        'tube_fit':range(1200,1216),'tube_calibration':range(1300,1316),'confirmation':range(2000,2024)}

def paths(split,k):
    b=BANK/split
    return b/f'{k}.world', b/f'{k}.json'

def sha_file(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def fingerprint(w):
    h=hashlib.sha256()
    components={}
    for attr in ('positions','noise','shadows'):
        arr=np.ascontiguousarray(getattr(w,attr))
        h.update(memoryview(arr));components[attr]=hashlib.sha256(memoryview(arr)).hexdigest()
    return h.hexdigest(),components

def generate(job):
    split,k=job
    dest,info=paths(split,k)
    if dest.is_file() and info.is_file():
        meta=json.loads(info.read_bytes())
        if sha_file(dest)!=meta['file_sha256']:raise ValueError(f'corrupt immutable world {dest}')
        return meta
    dest.parent.mkdir(parents=True,exist_ok=True)
    cfg=load_config();cfg['simulation'].update(warmup_s=0,service_s=180,sensing_budget=3/180)
    rng=np.random.default_rng(np.random.SeedSequence([20261127,k]))
    groups=[GROUPS[i] for i in rng.permutation(np.repeat(np.arange(6),[3,1,1,1,3,7]))]
    ns='g1_icc_server_v1/main/'+split
    w=World(cfg,ns,k,groups=groups)
    digest,components=fingerprint(w)
    tmp=dest.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:pickle.dump(w,stream,protocol=5)
    meta=dict(split=split,world=k,namespace=ns,group_seed=[20261127,k],groups=groups,
              physical_sha256=digest,components=components,file_sha256=sha_file(tmp),
              generator_sha256=sha_file(ENGINE/'routeclock/world.py'),numpy=np.__version__)
    tmp.replace(dest);dest.chmod(0o444);put(info,meta);info.chmod(0o444)
    return meta

def main():
    BANK.mkdir(parents=True,exist_ok=True)
    jobs=[(split,k) for split,ids in SPLITS.items() for k in ids]
    put(BANK/'plan.json',dict(splits={k:list(v) for k,v in SPLITS.items()},group_seed='[20261127,world_id]',counts=[3,1,1,1,3,7],
                                 template='package tools/run_n01.py world construction',confirmation_scores_read=False))
    with futures.ProcessPoolExecutor(30,mp_context=multiprocessing.get_context('fork')) as pool:
        m=list(pool.map(generate,jobs,chunksize=1))
    if len(set(r['physical_sha256'] for r in m))!=len(jobs):raise ValueError('duplicate physics across IDs')
    put(BANK/'index.json',dict(total=len(m),worlds=m))
    print('worlds',len(m),flush=True)

if __name__=='__main__':main()
