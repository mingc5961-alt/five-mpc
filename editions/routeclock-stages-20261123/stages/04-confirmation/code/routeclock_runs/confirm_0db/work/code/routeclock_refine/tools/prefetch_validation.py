"""Build the remaining full-size T validation worlds with one low-priority CPU."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routeclock.config import load_config
from refine.world_cache import paired_world

if __name__ == '__main__':
    cfg=load_config()
    cfg['simulation']['sensing_budget']=.05/3
    for idx in range(1,20):
        start=time.monotonic()
        world=paired_world(cfg,'refine_v1/validation',idx)
        print(json.dumps(dict(world=world.mission,wall_s=round(time.monotonic()-start,3))),flush=True)
