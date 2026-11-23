"""Compare native-query and grouped/batched inference on frozen real models."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import tabpfn_eval_fast as fast


def compare_csv(a, b):
    result = {}
    for name in ['forecast.csv', 'paired_forecast.csv', 'gate_events.csv', 'deadlines.csv']:
        left, right = pd.read_csv(a/name), pd.read_csv(b/name)
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-6, atol=1e-7)
        if name == 'deadlines.csv':
            for col in ['task', 'origin', 'method', 'head', 'oracle_s', 'approved_s', 'issued', 'early_failure', 'covered']:
                pd.testing.assert_series_equal(left[col], right[col], check_exact=True)
        result[name] = len(left)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--head', choices=['lightgbm', 'tabpfn'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--native-queries', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    path = args.run/f'five/{args.head}_7/bundle.pkl'
    indices = list(range(16))+[100, 167, 233, 300, 350, 351]
    b, anchors = fast.load_inputs(args.run, path, indices)
    anchors = anchors[:64]
    pairs = fast.Radio(b['config']).pairs
    report = dict(passed=False, bundle_sha256=fast.digest(path), runner_sha256=fast.digest(fast.__file__), anchors=len(anchors),
                  native_queries=args.native_queries,
                  tasks=len({a['task'] for a in anchors}), heads={},
                  tolerances=dict(rtol=1e-6, atol=1e-7, decision_columns='exact'))
    if args.head == 'lightgbm':
        fast.evaluate_anchors(anchors, b, args.out/'original')
    native, batched = {}, {}
    items = [(name, head, name) for name, head in b['heads'].items()]
    items += [('full', b['controls']['RC']['full'], 'RC'), ('duration', b['controls']['RC']['duration'], 'RC')]
    for name, head, method in items:
        f = (lambda part, mode: fast.action_values(head, part, pairs, mode)) if name == 'duration' else (
            lambda part, mode: fast.scale_values(head, part, method, mode))
        start = time.perf_counter()
        native[name] = f(anchors, False)
        native_s = time.perf_counter()-start
        start = time.perf_counter()
        batched[name] = f(anchors, not args.native_queries)
        batch_s = time.perf_counter()-start
        split = np.concatenate([f(anchors[:17], not args.native_queries), f(anchors[17:], not args.native_queries)])
        np.testing.assert_allclose(batched[name], native[name], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(split, native[name], rtol=1e-6, atol=1e-7)
        if args.head == 'tabpfn':
            head.reg.model = None  # The original loop recreates this context on head switches.
            reloaded = f(anchors[:1], False)
            np.testing.assert_allclose(reloaded, native[name][:1], rtol=1e-6, atol=1e-7)
            head.reg.model = None
            reloaded_last = f(anchors[-1:], False)
            np.testing.assert_allclose(reloaded_last, native[name][-1:], rtol=1e-6, atol=1e-7)
        report['heads'][name] = dict(native_s=native_s, batch_s=batch_s,
            max_absolute_difference=float(np.max(np.abs(batched[name]-native[name]))))
        fast.save_json(args.out/'validation.json', report)
        print(name, report['heads'][name], flush=True)
    fast.evaluate_anchors(anchors, fast.cached_bundle(b, anchors, native), args.out/'native_cached')
    fast.evaluate_anchors(anchors, fast.cached_bundle(b, anchors, batched), args.out/'batched_cached')
    report['csv_rows'] = compare_csv(args.out/'native_cached', args.out/'batched_cached')
    if args.head == 'lightgbm':
        compare_csv(args.out/'original', args.out/'batched_cached')
    else:
        fast.evaluate_anchors(anchors[:2], b, args.out/'original_two')
        fast.evaluate_anchors(anchors[:2], fast.cached_bundle(b, anchors, batched), args.out/'cached_two')
        compare_csv(args.out/'original_two', args.out/'cached_two')
    assert fast.digest(path) == report['bundle_sha256']
    report['passed'] = True
    fast.save_json(args.out/'validation.json', report)
    print('VALIDATION PASSED', args.head, flush=True)


if __name__ == '__main__':
    main()
