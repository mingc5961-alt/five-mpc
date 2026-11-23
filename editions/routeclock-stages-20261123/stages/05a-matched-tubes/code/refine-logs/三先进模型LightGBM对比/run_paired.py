"""Pair the published F/I/RC baselines with the 10 external-tube development worlds."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import os
import time

from refine import world_cache
from runtime import remaining_runs as runtime

root = Path('/data/chen/routeclock_runs/confirm_0db')
out = Path(__file__).resolve().parent
cache = out / 'cache'
cells = out / 'baseline_cells'
progress = out.parent / '实时实验进度.txt'
namespace = json.loads((root / 'scope.json').read_text())['analysis']['development_namespace']


def log(message):
    with progress.open('a') as handle:
        print(datetime.now().strftime('%F %T'), message, file=handle)


def paired(config, _namespace, world):
    return world_cache.paired_world(config, namespace, world)


def execute(item):
    world, budget, model = item
    path = cells / f'{world:03}_{budget}_{model}.json'
    if path.exists():
        data = json.loads(path.read_text())
        assert (data['namespace'], data['world'], data['budget'], data['model']) == (namespace, *item)
        return
    runtime.OUT = cache
    world_cache.BASE = root / 'work/baseline_cache'
    runtime.network.paired_world = paired
    row = dict(model=model, phase='development', world=world, condition=budget, mcs_margin_db=0,
               tube='default', risk_scale=1, max_interval=10)
    if model == 'F':
        row['fixed'] = dict(pair=[8, 32], period=3.0)
    start = time.monotonic()
    value = runtime.Runner().execute(row)
    data = dict(namespace=namespace, world=world, budget=budget, model=model,
                result=value, elapsed_s=time.monotonic()-start)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, allow_nan=False, default=lambda v:v.tolist()))
    temp.replace(path)


if __name__ == '__main__':
    from concurrent.futures import as_completed
    cache.mkdir(exist_ok=True)
    cells.mkdir(exist_ok=True)
    for name in ('F.json', 'classical.pkl'):
        link = cache / name
        if not link.exists():
            link.symlink_to(root / 'work/baseline_cache' / name)
    tasks = [(w,b,m) for m in ('I','RC','F') for b in 'TML' for w in range(10)]
    log('三先进模型 LightGBM 同世界对照开始。已核对开发集三模型 × 三档 × 十世界的 90 个外接管回执。补跑固定策略 F、IMM 和 RC 各 30 场，沿用冻结权重、0 dB 余量和原调度。开发世界预热缓存单独生成。')
    with ProcessPoolExecutor(8) as pool:
        futures = [pool.submit(execute, item) for item in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            if n % 15 == 0:
                msg=f'三先进模型 LightGBM 同世界对照：基线闭环回执 {n}/90；开发集外接管 90/90。'
                log(msg)
                print(msg, flush=True)
