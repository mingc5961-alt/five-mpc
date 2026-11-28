"""Measure warm CPU throughput for the restored Chronos seed-7 weights."""
import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path


def init_worker(base, weights, barrier):
    global torch, model, inputs, ready_barrier
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = Chronos2Pipeline.from_pretrained(str(base), device_map="cpu", dtype=torch.float32).model
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    t = torch.arange(360, dtype=torch.float32)
    context = torch.stack((t / 360, torch.sin(t / 19), torch.cos(t / 31)))
    mask = (t.long() % 6 == 5).expand_as(context)
    inputs = dict(context=context * mask, context_mask=mask, group_ids=torch.zeros(3, dtype=torch.long),
                  num_output_patches=2)
    with torch.inference_mode():
        model(**inputs)
    ready_barrier = barrier


def ready():
    ready_barrier.wait(timeout=240)
    return os.getpid()


def measure(calls):
    with torch.inference_mode():
        start = time.monotonic()
        for _ in range(calls):
            out = model(**inputs).quantile_preds
        checksum = float(out.sum())
        stop = time.monotonic()
    import resource
    return (os.getpid(), start, stop, checksum, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('/data/chen/routeclock_inputs_20261120/chronos'))
    p.add_argument('--weights', type=Path, default=root/'reference/models/Chronos/best.pt')
    p.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--calls', type=int, default=24)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.calls < 1 or min(a.levels) < 1 or a.output.exists():
        p.error('provide positive counts and an unused output path')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    ctx = mp.get_context('spawn')
    size = max(a.levels)
    rows = []
    cgroup = Path('/sys/fs/cgroup')
    baseline = int((cgroup/'memory.current').read_text())
    with cf.ProcessPoolExecutor(size, mp_context=ctx, initializer=init_worker,
                                initargs=(a.base, a.weights, ctx.Barrier(size))) as pool:
        warm = [pool.submit(ready) for _ in range(size)]
        assert len({f.result() for f in warm}) == size
        resident = int((cgroup/'memory.current').read_text())
        ref = None
        for levels in (a.levels, list(reversed(a.levels)), a.levels):
            for level in levels:
                jobs = [pool.submit(measure, a.calls) for _ in range(level)]
                records = [f.result() for f in jobs]
                assert len({r[0] for r in records}) == level
                if ref is None:
                    ref = records[0][3]
                if not all(math.isfinite(r[3]) and math.isclose(r[3], ref, abs_tol=1e-5) for r in records):
                    raise ValueError('forward checksum mismatch')
                elapsed = max(r[2] for r in records) - min(r[1] for r in records)
                rows.append(dict(workers=level, calls=level*a.calls, elapsed_s=elapsed,
                                 calls_s=level*a.calls/elapsed, max_worker_peak_rss_GiB=max(r[4] for r in records)))
    rates = {k:round(statistics.median(r['calls_s'] for r in rows if r['workers']==k),2) for k in a.levels}
    result = dict(scope='warm synthetic sparse XYZ prefix, one user per forward, 360 slots; CPU float32',
                  weights=str(a.weights), cpu_max=(cgroup/'cpu.max').read_text().strip(),
                  baseline_GiB=baseline/2**30, all_models_warm_GiB=resident/2**30,
                  pool_workers=size, checksum=ref, median_calls_s=rates, rounds=rows)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(rates)


if __name__ == '__main__':
    main()
