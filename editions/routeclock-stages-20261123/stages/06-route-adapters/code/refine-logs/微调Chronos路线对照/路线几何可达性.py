"""Project true endpoints onto the frozen, observed map for a hindsight diagnostic."""
import csv
import gzip
import pickle
from pathlib import Path

import numpy as np
from refine import network, world_cache
from routeclock.config import load_config
from routeclock.geometry import angles, angle_error

OUT = Path(__file__).resolve().parent
world_cache.BASE = Path('/data/chen/routeclock_runs/confirm_0db/work/baseline_cache')
HORIZONS = np.array([1, 3, 5, 10])

def closest(points, truth):
    start = points[:-1]
    delta = np.diff(points, axis=0)
    fraction = np.sum((truth[:, None] - start) * delta, axis=2)
    fraction /= np.maximum(np.sum(delta * delta, axis=1), 1e-20)
    candidates = start + fraction.clip(0, 1)[:, :, None] * delta
    distance = np.linalg.norm(candidates - truth[:, None], axis=2)
    return candidates[np.arange(len(truth)), distance.argmin(axis=1)]

def main():
    cfg = load_config()
    cfg['simulation']['sensing_budget'] = .05/3
    dest = OUT/'路线几何可达性.csv'
    temp = dest.with_suffix('.tmp')
    with temp.open('w') as stream:
        rows = csv.writer(stream)
        rows.writerow(['world','uid','horizon_s','geometry','position_m','angle_deg'])
        for w in range(40):
            world = world_cache.paired_world(cfg,'confirm_0db_20261122/v1',w)
            with gzip.open(OUT/'confirm_cache/warm'/f'v1_{w:03}.pkl.gz','rb') as fp:
                trackers = pickle.load(fp)
            assert len(trackers) == 16
            for uid, tr in enumerate(trackers):
                last = tr.history[-1].t
                state = tr.memory
                assert last == tr.snapshots[-1].origin and state.active is not None
                truth = network.interpolate(world.positions[uid],last+HORIZONS,world.dt)
                projected = np.stack([closest(f.points, truth) for f in state.fragments])
                errors = np.linalg.norm(projected-truth,axis=2)
                all_best = projected[errors.argmin(axis=0),np.arange(len(truth))]
                neighbors = [state.active, *(dst for src,dst in state.connections if src == state.active)]
                ids = np.asarray(neighbors)
                next_best = projected[ids[errors[ids].argmin(axis=0)],np.arange(len(truth))]
                for method,pred in [('current_fragment',projected[state.active]),
                                    ('at_most_one_observed_transition',next_best),('all_fragments',all_best)]:
                    dist = np.linalg.norm(pred-truth,axis=1)
                    angular = np.rad2deg(np.linalg.norm(angle_error(angles(pred),angles(truth)),axis=1))
                    rows.writerows((w,uid,h,method,m,a) for h,m,a in zip(HORIZONS,dist,angular,strict=True))
    temp.replace(dest)
if __name__ == '__main__':main()
