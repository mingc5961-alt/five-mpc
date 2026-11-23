"""Recover causal observations for the archived synthetic route splits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import numpy as np
from routeclock.config import load_config
from routeclock.dataset import observation_schedule
from routeclock.world import World, rng_for

BASE = Path('/data/chen/routeclock_runs/refine_v1')
ASSETS = Path('/data/chen/routeclock_inputs_20261120')
ARCHIVE = Path('/data/chen/formal_execution/run_20261117_001732/five_data')
ROOT = Path(__file__).resolve().parents[1]

def prepare(job):
    split, index = job
    dest = BASE/'observations'/split/f'{index:04}.npz'
    if dest.exists():
        return
    cfg = load_config()
    namespace = f'offline/{split}'
    with (ARCHIVE/split/f'{index:04}.pkl').open('rb') as stream:
        rows = pickle.load(stream)
    world = World(cfg, namespace, index, n=1, groups=[rows[0]['group']])
    rng = rng_for(namespace, index, 777)
    candidates = [rng.uniform(low, high) for low, high in cfg['data']['anchor_intervals_s']]
    selected = [float(rng.choice(candidates))] if split == 'calibration' else candidates
    clock = np.array(observation_schedule(candidates, selected, [.5,1,2,5,10,None][index%6], rng))
    obs = [world.observe(0,t) for t in clock]
    origins = [row['package'].origin for row in rows]
    np.testing.assert_allclose(sorted(selected), origins, atol=1e-9, rtol=0)
    for row in rows:
        t = row['package'].origin
        np.testing.assert_allclose(world.truth(0,t)[0], row['truth'][0], atol=1e-7, rtol=0)
    tmp = dest.with_suffix('.part')
    with tmp.open('wb') as f:
        np.savez_compressed(f, times=clock, positions=[o.position for o in obs],
                            cov=[o.covariance for o in obs], radial=[o.radial_velocity for o in obs],
                            radial_std=[o.radial_std for o in obs], origins=origins,
                            truth=np.stack([r['truth'] for r in rows]))
    tmp.replace(dest)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    counts = {'train':1200,'validation':400,'calibration':400}
    jobs = []
    for split,n in counts.items():
        (BASE/'observations'/split).mkdir(exist_ok=True,parents=True)
        jobs.extend((split,i) for i in range(n))
    # Interleave route types and splits so small development batches become available early.
    jobs.sort(key=lambda item: (item[1]%50,item[1]//50,item[0]))
    with ProcessPoolExecutor(args.workers) as pool:
        for n,_ in enumerate(pool.map(prepare,jobs,chunksize=1),1):
            if n%20==0: print(f'{n}/{len(jobs)}',flush=True)
