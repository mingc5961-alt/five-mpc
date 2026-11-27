import sys,pickle,os
import numpy as np
from routeclock.world import World,GROUPS
from routeclock.config import load_config
from base import *
def generate(w):
    p=worldfile(w)
    if p.exists():return
    cfg=load_config()
    # Main 8/8 tests, and independent worlds for the controlled training-mixture study.
    counts=[2,1,1,1,3,8] if w<2000 else [4,3,3,2,2,2]
    # Paired arms in R09 share the namespace and absolute-time noise generator.
    seed=w-3000 if w>=3000 else (w if w<2000 else w-1000)
    if w>=3000:counts=[2,1,1,1,3,8]
    group=np.repeat(GROUPS,counts);group=np.random.default_rng([20261125,seed]).permutation(group).tolist()
    if w>=3000:
        group=[GROUPS[-1] if g==GROUPS[0] else g for g in group];counts=[0,1,1,1,3,10]
    world=World(cfg,'formal_8x8_nov25_v1',seed,groups=group)
    assert world.n==16 and cfg['simulation']['service_s']==180
    tmp=p.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:pickle.dump(world,f,protocol=5)
    tmp.replace(p)
    js(p.with_suffix('.json'),dict(world=w,seed=seed,namespace=world.namespace,counts=counts,groups=group,sha256=sha(p),events=world.detour_windows))
if __name__=='__main__':generate(int(sys.argv[1]))
