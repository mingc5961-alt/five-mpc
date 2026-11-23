"""Build independent common warm starts for the fixed-slot scheduling experiment."""
import os
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from routeclock.config import load_config
from routeclock.world import World
from routeclock.predictors import Tracker

OUT = ROOT / "runs" / "fixed_slots_cpu"
COST, QUANTUM, N, ROUNDS, SLOTS = .0031, .01, 16, 2, 64
PERIOD = COST + QUANTUM * N * ROUNDS

def prepare(key):
    split, mission = key
    path = OUT / f"{split}_{mission}.pkl"
    if path.exists():
        return
    cfg = load_config()
    cfg["simulation"]["service_s"] = SLOTS * PERIOD
    world = World(cfg, "network/fixed_slot_v1/" + split, mission, n=N)
    trackers = []
    for uid in range(N):
        tr = Tracker(cfg)
        for t in np.arange(0., 180., 2.) + uid * .125 + COST:
            tr.observe(world.observe(uid, t))
            # The map and IMM see every observation. Replay uses only 60 s
            # of resolved errors from forecasts issued up to 10 s earlier.
            if t >= 108:
                tr.issue()
                tr.snapshots.clear()  # Only Chronos consumes the snapshot list.
        trackers.append(tr)
    with path.with_suffix(".tmp").open("wb") as stream:
        pickle.dump((world, trackers), stream, protocol=5)
    path.with_suffix(".tmp").replace(path)
    return key

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for key in pool.map(prepare, [(s,i) for s,k in [("train",4),("test",6)] for i in range(k)]):
            print("prepared",key, flush=True)
