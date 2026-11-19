"""Reuse deterministic worlds and issue-time snapshots across paired policies."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time
from functools import lru_cache

PROJECT=Path(__file__).resolve().parents[1]
CACHE=Path('/data/chen/routeclock_runtime_cache/paired600')
LIMIT=6*1024**3


@lru_cache(None)
def file_hash(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


@lru_cache(None)
def version():
    files=[*sorted((PROJECT/'routeclock').glob('*.py')),PROJECT/'tools/ch6_experiment.py',Path(__file__)]
    return ''.join(file_hash(path) for path in files)


def prune(root):
    with (root/'trim.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        entries=[]
        for p in root.glob('*.pkl.gz'):
            try:s=p.stat()
            except FileNotFoundError:continue
            entries.append((s.st_mtime,p,s.st_size))
        total=sum(e[2] for e in entries)
        for _,p,size in sorted(entries):
            if total<=LIMIT:break
            with p.with_suffix('.lock').open('a') as item:
                try:fcntl.flock(item,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                p.unlink(missing_ok=True);total-=size


def snapshot(kind,key,build,root):
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    token=hashlib.sha256(json.dumps([version(),key],sort_keys=True).encode()).hexdigest()
    p=root/f'{kind}-{token}.pkl.gz'
    start=time.monotonic()
    with p.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if p.exists():
            try:
                with gzip.open(p,'rb') as f:
                    value=pickle.load(f)
                    if f.read(1):raise ValueError('unexpected cache trailer')
                p.touch()
                return value,dict(hit=True,seconds=time.monotonic()-start)
            except (EOFError,OSError,pickle.UnpicklingError,ValueError):
                p.unlink(missing_ok=True)
        value=build()
        if shutil.disk_usage(root).free>2*1024**3:
            with tempfile.NamedTemporaryFile(dir=root,delete=False) as f: tmp=Path(f.name)
            try:
                with gzip.open(tmp,'wb',compresslevel=1) as f:pickle.dump(value,f,protocol=5)
                os.replace(tmp,p)
            finally:tmp.unlink(missing_ok=True)
    prune(root)
    return value,dict(hit=False,seconds=time.monotonic()-start)


def prepare(cfg,namespace,mission,cost,warm_budget,adapter,expert,prewarm,root=CACHE):
    from routeclock.world import World
    # Only these two service-only settings vary between the paired arms.
    # Warmup cost and budget enter the separate tracker key below.
    sim=dict(cfg['simulation'])
    del sim['observation_cost_s'],sim['sensing_budget']
    key=[namespace,mission,{**cfg,'simulation':sim}]
    w,world_time=snapshot('world',key,lambda:World(cfg,namespace,mission),root)
    w.cfg=cfg
    checkpoint=None
    if adapter is not None:
        ckpt=Path(adapter.checkpoint)
        checkpoint=[str(ckpt),[(x.name,file_hash(x)) for x in sorted(ckpt.iterdir()) if x.is_file()],
                    adapter.explicit_route_mask]
    def build():
        trackers=prewarm(cfg,w,cost,adapter,expert,warm_budget)
        for t in trackers:t.chronos=None
        return trackers
    ts,warm_time=snapshot('warm',[key,cost,warm_budget,expert,checkpoint],build,root)
    for t in ts:
        t.cfg=cfg
        t.chronos=adapter
    return w,ts,dict(world=world_time,warm=warm_time)
