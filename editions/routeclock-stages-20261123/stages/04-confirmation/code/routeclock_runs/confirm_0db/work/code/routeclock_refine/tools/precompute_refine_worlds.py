"""Build shared worlds ahead of confirmation workers, in their consumption order."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from refine.world_cache import paired_world
from refine.data import BASE
from routeclock.config import load_config

def build(i):
    cfg=load_config()
    cfg['simulation']['sensing_budget']=.05/3
    start=time.monotonic()
    world=paired_world(cfg,'refine_v1/confirmation',i)
    return dict(world=world.mission,wall_s=round(time.monotonic()-start,3),at=time.time())

if __name__=='__main__':
    workers=int(sys.argv[1])
    ids=[v for i in range(35) for v in (i,i+35)]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('fork')) as ex:
        for r in ex.map(build,ids,chunksize=1):
            print(json.dumps(r),flush=True)
