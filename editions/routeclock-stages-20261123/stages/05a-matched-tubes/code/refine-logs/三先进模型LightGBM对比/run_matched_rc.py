"""Score RC with its previously fitted 0.90-quantile head on the same development worlds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import pickle
import time
from unittest.mock import patch
import numpy as np
from run_paired import cache,cells,log,namespace,out,root,runtime,world_cache
from refine import network,tubes
from routeclock.config import load_config
from routeclock.radio import Radio

bundle_path = out.parent / 'RC_Chronos2_差距定位/跨版本归因/RC_matched/bundle.pkl'


def quantile_width(tree, features):
    flat = features.reshape(-1, features.shape[-1])
    return np.maximum(tubes.EPS,tree.predict(flat,num_threads=1).reshape(-1,21,2))


def execute(item):
    world_id, budget = item
    path = cells / f'{world_id:03}_{budget}_RCq90.json'
    if path.exists():
        return
    start = time.monotonic()
    runtime.OUT = cache
    world_cache.BASE = root / 'work/baseline_cache'
    cfg = load_config()
    cfg['simulation']['sensing_budget'] = runtime.BUDGET['T']
    world = world_cache.paired_world(cfg,namespace,world_id)
    cfg = load_config()
    cfg['simulation']['sensing_budget'] = runtime.BUDGET[budget]
    original = pickle.loads((cache/'classical.pkl').read_bytes())['RC']
    ext = pickle.loads(bundle_path.read_bytes())
    model = runtime.CausalModel('RC',original,cfg)
    trackers = runtime.warm_trackers(world)
    first = model.batch
    def init(histories):
        for history,tracker in zip(histories,trackers,strict=True):
            np.testing.assert_array_equal([o.position for o in history],[o.position for o in tracker.history])
            model.trackers[id(history)] = tracker
        model.batch = first
        return first(histories)
    model.batch = init
    radio = Radio(cfg)
    radio.threshold -= 5
    with runtime.controller(radio,10.), patch.object(network,'tree_scale',quantile_width):
        value,_ = network.run_world(cfg,world,model,dict(tree=ext['tree'],factors={'lightgbm':ext['factors']}),'lightgbm')
    data = dict(namespace=namespace,world=world_id,budget=budget,model='RCq90',tube='lightgbm',
                bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                result=value,elapsed_s=time.monotonic()-start)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,allow_nan=False))
    temp.replace(path)


if __name__ == '__main__':
    log('为分清主线 RC 原装管与同配方比较，加入 RC 同规格 0.90 分位 LightGBM × 三档 × 十世界，共 30 场。使用前次已拟合的 RC 专属误差头；与本次基线共用开发集预热，保留同一中心和门控。')
    with ProcessPoolExecutor(3) as pool:
        jobs=[pool.submit(execute,(w,b)) for b in 'TML' for w in range(10)]
        for n,j in enumerate(as_completed(jobs),1):
            j.result()
            if n%10==0:
                log(f'RC 同规格 LightGBM 闭环已完成 {n}/30。')
