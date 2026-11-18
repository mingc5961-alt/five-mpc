"""Resume evaluation of a frozen bundle; cache model-grouped inference only.

The original evaluate_anchors function still computes every scientific metric.
No fitted parameter, calibration factor, sample, or action rule is changed.
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SOURCE = Path('/data/chen/routeclock_server/routeclock_five_20261115')
os.environ.setdefault('ROUTECLOCK_TABPFN_CHECKPOINT', str(SOURCE.parent/'weights/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
sys.path.insert(0, str(SOURCE))
from routeclock.five import evaluate_anchors
from routeclock.training import inject_gate
from routeclock.validity import feature_rows, QUERY_H, EPS, Regressor
from routeclock.radio import Radio


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def log(out, stage, **fields):
    row = dict(at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), stage=stage, **fields)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    save_json(out/'eval_progress.json', row)
    with (out/'eval_progress.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')


def load_inputs(run, bundle_path, indices=None):
    with bundle_path.open('rb') as f:
        bundle = pickle.load(f)
    assert bundle['counts'] == [1200, 400, 400, 400]
    anchors = []
    for i in range(400) if indices is None else indices:
        with (run/'five_data/test'/f'{i:04}.pkl').open('rb') as f:
            anchors.extend(pickle.load(f))
    inject_gate(anchors, bundle['gates']['RC'], 'RC', 'P')
    for head in [*bundle['heads'].values(), *bundle['controls']['RC'].values()]:
        if hasattr(head, 'reg'):
            head.reg.device = 'cuda'
    return bundle, anchors


def scale_values(head, anchors, method, batched):
    if not batched:
        return np.stack([head.predict(a['package'], a['features'][method]) for a in anchors])
    rows = np.concatenate([feature_rows(a['features'][method], a['package'].hs, mode=head.mode) for a in anchors])
    pred = head.reg.predict(rows).reshape(len(anchors), len(QUERY_H), 2)
    raw = np.maximum.accumulate(np.maximum(np.exp(np.clip(pred, -30, 5))-EPS, EPS), axis=1)
    env = raw*head.factor
    return np.stack([e[np.searchsorted(QUERY_H, a['package'].hs, side='left')] for e, a in zip(env, anchors)])


def action_values(head, anchors, pairs, batched):
    matrices = [head.rows(a['features']['RC'], a['package'].hs, pair, [0]) for a in anchors for pair in pairs]
    if batched:
        return head.lower(np.concatenate(matrices)).reshape(len(anchors), len(pairs))
    return np.asarray([float(head.lower(x)[0]) for x in matrices]).reshape(len(anchors), len(pairs))


class CachedScale:
    def __init__(self, anchors, values):
        self.values = {id(a['package']): value for a, value in zip(anchors, values)}

    def predict(self, package, features):
        return self.values[id(package)]


class CachedAction:
    def __init__(self, original, anchors, pairs, values):
        self.original = original
        self.target = original.target
        self.values = {}
        for a, row in zip(anchors, values):
            for pair, value in zip(pairs, row):
                key = self.key(original.rows(a['features']['RC'], a['package'].hs, pair, [0]))
                if key in self.values:
                    np.testing.assert_allclose(self.values[key], value, rtol=1e-6, atol=1e-7)
                self.values[key] = value

    @staticmethod
    def key(rows):
        x = np.ascontiguousarray(rows)
        return (x.shape, x.dtype.str, x.tobytes())

    def rows(self, *args, **kwargs):
        return self.original.rows(*args, **kwargs)

    def lower(self, rows):
        return np.array([self.values[self.key(rows)]])


def cached_bundle(bundle, anchors, cache):
    result = dict(bundle, heads=dict(bundle['heads']), controls={'RC':dict(bundle['controls']['RC'])})
    for method in bundle['heads']:
        result['heads'][method] = CachedScale(anchors, cache[method])
    result['controls']['RC']['full'] = CachedScale(anchors, cache['full'])
    result['controls']['RC']['duration'] = CachedAction(bundle['controls']['RC']['duration'], anchors,
                                                      Radio(bundle['config']).pairs, cache['duration'])
    return result


def compute_cache(bundle, anchors, out, batch_size=64, batched=True):
    folder = out/'inference_cache'
    folder.mkdir(exist_ok=True)
    cache = {}
    heads = [(name, head, name) for name, head in bundle['heads'].items()]
    heads += [('full', bundle['controls']['RC']['full'], 'RC'),
              ('duration', bundle['controls']['RC']['duration'], 'RC')]
    pairs = Radio(bundle['config']).pairs
    for name, head, method in heads:
        path = folder/f'{name}.npz'
        values = []
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                values = list(data['values'])
            assert len(values) <= len(anchors)
        log(out, 'inference', head=name, anchors_done=len(values), anchors_total=len(anchors))
        for start in range(len(values), len(anchors), batch_size):
            part = anchors[start:start+batch_size]
            result = (action_values(head, part, pairs, batched) if name == 'duration'
                      else scale_values(head, part, method, batched))
            assert np.isfinite(result).all()
            values.extend(result)
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f, values=np.asarray(values))
            temp.replace(path)
            log(out, 'inference', head=name, anchors_done=len(values), anchors_total=len(anchors))
        cache[name] = np.asarray(values)
        head.reg.model = None
        Regressor._active_tabpfn = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-bundle-sha256', required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--native-queries', action='store_true')
    args = parser.parse_args()
    assert args.batch_size > 0
    assert digest(args.bundle) == args.expected_bundle_sha256, 'Frozen bundle changed'
    validation = json.loads(args.validation.read_text())
    assert validation['passed'] and validation['bundle_sha256'] == args.expected_bundle_sha256
    assert validation['runner_sha256'] == digest(__file__), 'Validated runner changed'
    assert validation['native_queries'] == args.native_queries, 'Unvalidated query mode'
    import torch
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    binding = dict(bundle_sha256=args.expected_bundle_sha256, runner_sha256=digest(__file__),
                   batch_size=args.batch_size, batched=not args.native_queries,
                   test_files={p.name:digest(p) for p in sorted((args.run/'five_data/test').glob('*.pkl'))})
    binding_path = args.out/'eval_cache_binding.json'
    if binding_path.exists():
        assert json.loads(binding_path.read_text()) == binding, 'Evaluation cache identity mismatch'
    else:
        save_json(binding_path, binding)
    bundle, anchors = load_inputs(args.run, args.bundle)
    assert (bundle['kind'], bundle['seed']) == ('tabpfn', 7)
    assert len(anchors) == 1200 and len({a['task'] for a in anchors}) == 400
    cache = compute_cache(bundle, anchors, args.out, args.batch_size, not args.native_queries)
    log(args.out, 'original_metric_evaluation', anchors=len(anchors), tasks=400)
    evaluate_anchors(anchors, cached_bundle(bundle, anchors, cache), args.out)
    assert digest(args.bundle) == args.expected_bundle_sha256
    log(args.out, 'complete', anchors=len(anchors), tasks=400, bundle_sha256=args.expected_bundle_sha256)


if __name__ == '__main__':
    main()
