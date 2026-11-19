"""Independent long-history routes and shared deployment histories for the pilot."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from routeclock.config import load_config, dump_json
from routeclock.world import World, GROUPS
from routeclock.predictors import Tracker

OUT = ROOT / "runs/long_history_deadline_pilot"
CENTERS = ("I", "P")
KINDS = ("envelope", "lightgbm", "tabpfn")
HORIZONS = np.arange(501) * .02


def configuration():
    cfg = load_config()
    cfg["simulation"].update(primary_N=3, warmup_s=180, service_s=18)
    return cfg


def generate(job):
    split, index = job
    path = OUT / f"{split}_{index:02}.pkl"
    if path.exists():
        return str(path.name)
    network = split == "network"
    groups = [GROUPS[(3*index+i) % 6] for i in range(3)] if network else [GROUPS[index % 6]]
    cfg = configuration()
    world = World(cfg, f"long-history-deadline-v1/{split}", index, n=len(groups), groups=groups)
    records, histories = [], []
    for i, group in enumerate(groups):
        tracker = Tracker(cfg)
        # Fixed independent-of-method warmup. Last observations staggered before service.
        times = np.arange(180, dtype=float) + .5 + .1*i
        for t in times:
            tracker.observe(world.observe(i, t))
            # Forecasts issued >70s before the query cannot enter its 60s archive:
            # maximum forecast lifetime is 10s. State and map update at every observation.
            if t >= times[-1] - 71:
                package = tracker.issue()
        origin = package.origin
        truth = np.asarray([world.truth(i, origin+h)[0] for h in HORIZONS])
        records.append(dict(task=f"{split}/{index}/{i}", group=group, package=package,
                            features={m: package.features for m in CENTERS}, truth_h=HORIZONS,
                            truth=truth, route_hash=hashlib.sha256(world.references[i].tobytes()).hexdigest()))
        tracker.snapshots = [package]  # Neither center uses Chronos historical covariates.
        if network:
            histories.append(tracker)
    with path.with_suffix(".tmp").open("wb") as f:
        pickle.dump((records, histories, world if network else None), f)
    path.with_suffix(".tmp").replace(path)
    return path.name


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"train":24, "cal":24, "network":4}
    dump_json(OUT/"design.json", dict(config=configuration(), counts=counts, centers=CENTERS, heads=KINDS,
              nominal_alpha=.1, comparable_failure_absolute_percentage_points=2,
              narrowest_pair=[32,32], evaluation="four paired network missions, six behaviors twice",
              geometric_failure="actual directional gain loss >3dB; also report unavailable airtime",
              observation_protocol="180 warmup observations per vehicle, 1s interval; then shared value scheduler",
              training="all 8 horizons and both axes; fixed LightGBM params; no test-set tuning"))
    jobs = [(split,i) for split,n in counts.items() for i in range(n)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for result in pool.map(generate, jobs):
            print(result, flush=True)
