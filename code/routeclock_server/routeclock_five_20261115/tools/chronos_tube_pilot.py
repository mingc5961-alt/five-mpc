"""Paired CPU pilot using native Chronos quantile widths and the frozen LightGBM tube."""
import argparse
import json
import pickle
from pathlib import Path

from pilot_histories import ROOT, OUT, configuration
import numpy as np
import pandas as pd
import torch
from routeclock.config import dump_json
from routeclock.foundation import ChronosAdapter
from routeclock.geometry import angular_jacobian, angles
from routeclock.radio import Radio, array_gain
from routeclock.scheduler import frontier
from routeclock.simulator import simulate
from routeclock.validity import ScaleHead, QUERY_H, EPS, interval_errors


class QuantileTube(ScaleHead):
    """Propagate marginal quantile spreads into angular scales before split calibration."""
    def __init__(self):
        self.factor = None

    def raw(self, package, features):
        positions = package.centers['C']
        jacobian = np.array([angular_jacobian(p) for p in positions])
        variance = np.einsum('hai,hij,haj->ha', jacobian, package.covariance['C'], jacobian)
        scale = np.sqrt(np.maximum(variance, 0))
        cumulative = np.maximum.accumulate(np.maximum(scale, EPS), axis=0)
        return cumulative[np.searchsorted(package.hs, QUERY_H)]


def read_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/chronos_tube_cpu')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = configuration()
    old = OUT/'results'
    manifest = json.loads((old/'manifest.json').read_text())
    model = ChronosAdapter('cpu', str((ROOT/manifest['checkpoint']).resolve()))
    cal = [read_pickle(path)[0][0] for path in sorted(OUT.glob('cal_*.pkl'))]
    networks = [read_pickle(path) for path in sorted(OUT.glob('network_*.pkl'))]
    test = [a for records, _, _ in networks for a in records]
    anchors = cal + test
    assert len(cal) == 12 and len(test) == 6 and len({a['route_hash'] for a in anchors}) == 18
    for start in range(0, len(anchors), 4):
        batch = anchors[start:start + 4]
        items = []
        for a in batch:
            inp, offset = a['inputs'][0]
            past = inp['past_covariates']
            cov = np.array([[past[f'position_cov_{min(i,j)}{max(i,j)}'][-1]
                             for j in range(3)] for i in range(3)])
            items.append((inp, offset, cov))
        for a, (pos, cov) in zip(batch, model.predict_inputs(items), strict=True):
            a['package'].centers['C'] = pos
            a['package'].covariance['C'] = cov
    bundle = read_pickle(old/'bundle.pkl')
    lgb = bundle['heads']['C']
    native = QuantileTube().calibrate(cal, 'C')
    dump_json(args.out/'calibration.json', dict(alpha=.1, factor=native.factor,
              scores=native.calibration_scores, routes=[a['route_hash'] for a in cal]))
    radio = Radio(cfg)
    rows = []
    for a in test:
        pkg = a['package']
        error = interval_errors(pkg, 'C', a['truth_h'], a['truth'])
        pointing = angles(np.array([pkg.at('C', h) for h in a['truth_h']]))
        actual_loss = -10*np.log10(array_gain(pointing, angles(a['truth']), (32,32))/1024)
        for name, head in [('lightgbm', lgb), ('native', native)]:
            env = head.predict(pkg, a['features']['C'])
            actions = frontier(pkg, 'C', env, radio, cfg)
            approved = max((act.interval for act in actions if act.pair == (32,32)), default=0.)
            rows.append(dict(task=a['task'], head=name, approved_s=approved,
                             covered=bool(np.all(error <= env[np.searchsorted(pkg.hs, QUERY_H)])),
                             early_failure=bool(approved and np.any(actual_loss[a['truth_h'] <= approved]>3))))
    pd.DataFrame(rows).to_csv(args.out/'anchors.csv', index=False)
    previous = pd.read_csv(old/'deadlines.csv').query("method=='C' and head=='common_local'").set_index('task')
    measured = pd.DataFrame(rows).query("head=='lightgbm'").set_index('task')
    np.testing.assert_equal(measured.approved_s.values, previous.loc[measured.index].approved_s.values)
    reference = pd.read_csv(old/'closed_loop.csv')
    results = reference.query("method in ['C','RC'] and action_mode=='joint'").copy()
    results['tube'] = 'lightgbm'
    bundle['heads']['C'] = native
    bundle['kind'] = 'native_quantiles'
    bundle['action_reference'] = json.loads((ROOT/'runs/fixed_validation/winner.json').read_text())
    for mission,(records, trackers, world) in enumerate(networks):
        archived = json.loads((old/f'loop_{mission}_C_joint/predictions.json').read_text())
        for i,(tracker,a) in enumerate(zip(trackers, records, strict=True)):
            assert tracker.snapshots[-1] is a['package']
            np.testing.assert_allclose(a['package'].centers['C'], archived[i]['center'], atol=.001, rtol=0)
        result = simulate(cfg, world, bundle, 'C', model, args.out/f'loop_{mission}', service_trackers=trackers)
        result = {k:v for k,v in result.items() if not isinstance(v,(dict,list,np.ndarray))}
        result['tube'] = 'native'
        results = pd.concat([results,pd.DataFrame([result])], ignore_index=True)
        results.to_csv(args.out/'closed_loop.csv', index=False)
        print(mission, result['net_mbps'], result['narrow_share'], result['hold_failures'], flush=True)
    summarize(results, args.out)


def summarize(results, out):
    rows = []
    for (method,tube), group in results.groupby(['method','tube']):
        assert len(group) == 2
        assert np.allclose(group.accounted_s, group.service_s)
        seconds = (group.narrow_hold_s.fillna(0)*group.narrow_segments).sum()
        np.testing.assert_allclose(seconds, (group.narrow_share*group.service_s*group.N).sum())
        segments = group.narrow_segments.sum()
        rows.append(dict(method=method,tube=tube,net_mbps=group.net_mbps.mean(),
                         observations=group.observations_service.mean(),narrow_uav_seconds=seconds,
                         narrow_share=group.narrow_share.mean(), narrow_segments=segments,
                         narrow_hold_s=seconds/segments if segments else None,
                         narrow_failures=group.narrow_failures.sum(), failures=group.hold_failures.sum(),
                         segments=group.hold_segments.sum()))
    summary = pd.DataFrame(rows)
    gate = summary.loc[summary.method=='RC','net_mbps'].item()
    summary['gap_to_gate_mbps'] = gate-summary.net_mbps
    summary.to_csv(out/'summary.csv', index=False)


if __name__ == '__main__':
    main()
