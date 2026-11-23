"""Shared immutable arrays for paired worlds, validated by input fingerprint."""
import fcntl
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from routeclock.world import World
from .data import BASE,ROOT

ARRAYS=('positions','velocities','noise','shadows')
def paired_world(cfg,namespace,mission):
    physics=[(ROOT/'routeclock'/f).read_bytes() for f in ('world.py','geometry.py')]
    key=hashlib.sha256(b''.join(physics)+json.dumps([cfg,namespace,mission],sort_keys=True).encode()).hexdigest()
    folder=BASE/'world_cache';folder.mkdir(exist_ok=True)
    dest=folder/key
    with (folder/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not dest.exists():
            if shutil.disk_usage(folder).free<14*1024**3:return World(cfg,namespace,mission)
            world=World(cfg,namespace,mission)
            temp=Path(tempfile.mkdtemp(dir=folder))
            try:
                for attr in ARRAYS:np.save(temp/(attr+'.npy'),np.asarray(getattr(world,attr)))
                (temp/'metadata.json').write_text(json.dumps(dict(n=world.n,groups=world.groups,dt=world.dt,samples=len(world.times))))
                temp.rename(dest)
            finally:
                if temp.exists():shutil.rmtree(temp)
    meta=json.loads((dest/'metadata.json').read_text())
    world=World.__new__(World)
    world.cfg=cfg;world.n=meta['n'];world.groups=meta['groups'];world.dt=meta['dt']
    world.namespace=namespace;world.mission=mission;world.times=np.arange(meta['samples'])*world.dt
    for attr in ARRAYS:setattr(world,attr,np.load(dest/(attr+'.npy'),mmap_mode='r'))
    return world
