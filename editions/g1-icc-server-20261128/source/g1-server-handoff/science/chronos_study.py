"""Independently frozen original-weight Chronos baseline for the new Linux study."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
from server_paths import RUN
from closed import freeze
from replay_legacy import run

def read(split,k,m):
    return RUN/'native'/f'{split}_{k}_C_{m:.2f}.json'


def batch(split,worlds,scales,workers):
    jobs=[(split,k,'C',m) for k in worlds for m in scales if not read(split,k,m).exists()]
    with ProcessPoolExecutor(workers) as pool:
        fs=[pool.submit(run,*j) for j in jobs]
        for i,f in enumerate(as_completed(fs),1):
            f.result();print(split,i,len(fs),flush=True)

if __name__=='__main__':
    dev=range(1400,1408)
    # Native comparison family already triggered the shared development extension.
    batch('development',dev,[.30,1.25],8)
    scales=(.30,.45,.60,.75,1.,1.25)
    paths=[read('development',k,m) for k in dev for m in scales]
    while not all(p.exists() for p in paths):time.sleep(5)
    p=freeze({m:[json.loads(read('development',k,m).read_bytes()) for k in dev] for m in scales})
    (RUN/'chronos_freeze.json').write_text(json.dumps(p,indent=2))
    batch('rebuild_holdout',range(4000,4024),sorted(set(p['weights'])|{.75}),24)
