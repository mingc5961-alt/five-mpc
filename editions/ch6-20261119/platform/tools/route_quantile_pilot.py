"""Complete the two-by-two local comparison of Chronos and the frozen route gate."""
import json
import pickle
import pandas as pd
import numpy as np

from pilot_histories import ROOT, OUT, configuration
from routeclock.config import dump_json
from routeclock.predictors import apply_gate
from routeclock.route_quantile import RouteQuantileTube
from routeclock.validity import QUERY_H, interval_errors
from routeclock.geometry import angles
from routeclock.radio import Radio, array_gain
from routeclock.scheduler import frontier
from routeclock.simulator import simulate


def load(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def main():
    old = OUT/'results'
    chronos = ROOT/'runs/chronos_tube_cpu'
    out = ROOT/'runs/route_gate_quantile_cpu'
    out.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    radio = Radio(cfg)
    bundle = load(old/'bundle.pkl')
    cal = [load(p)[0][0] for p in sorted(OUT.glob('cal_*.pkl'))]
    networks = [load(p) for p in sorted(OUT.glob('network_*.pkl'))]
    test = [a for records,_,_ in networks for a in records]
    assert [a['route_hash'] for a in cal] == json.loads((chronos/'calibration.json').read_text())['routes']
    assert len(test)==6 and len({a['route_hash'] for a in cal+test})==18
    for a in cal+test:
        a['features']['RC'] = a['features']['P']
        apply_gate(a['package'], bundle['gates']['RC'], a['features']['P'], 'RC','P')
    head = RouteQuantileTube().calibrate(cal,'RC')
    dump_json(out/'calibration.json', dict(factor=head.factor, alpha=.1, scores=head.calibration_scores,
              routes=[a['route_hash'] for a in cal]))
    anchors = []
    for a in test:
        pkg = a['package']
        errors = interval_errors(pkg,'RC',a['truth_h'],a['truth'])
        pred = np.array([pkg.at('RC',t) for t in a['truth_h']])
        loss = -10*np.log10(array_gain(angles(pred),angles(a['truth']),(32,32))/1024)
        for name,model in [('lightgbm',bundle['heads']['RC']),('native',head)]:
            env = model.predict(pkg,a['features']['P'])
            choices = frontier(pkg,'RC',env,radio,cfg)
            approved = max((x.interval for x in choices if x.pair==(32,32)),default=0.)
            anchors.append(dict(task=a['task'],head=name,approved_s=approved,
                covered=bool(np.all(errors<=env[np.searchsorted(pkg.hs,QUERY_H)])),
                early_failure=bool(approved and np.any(loss[a['truth_h']<=approved]>3))))
    anchor_df=pd.DataFrame(anchors)
    reference=pd.read_csv(old/'deadlines.csv').query("method=='RC' and head=='common_local'").set_index('task')
    baseline=anchor_df.query("head=='lightgbm'").set_index('task')
    np.testing.assert_array_equal(baseline.approved_s,reference.loc[baseline.index].approved_s)
    anchor_df.to_csv(out/'anchors.csv',index=False)
    results=pd.read_csv(chronos/'closed_loop.csv')
    assert len(results)==6
    bundle['heads']['RC']=head
    bundle['kind']='route_mixture_quantiles'
    bundle['action_reference']=json.loads((ROOT/'runs/fixed_validation/winner.json').read_text())
    for mission,(records,trackers,world) in enumerate(networks):
        saved=json.loads((old/f'loop_{mission}_RC_joint/predictions.json').read_text())
        for i,a in enumerate(records):
            assert trackers[i].snapshots[-1] is a['package']
            np.testing.assert_allclose(a['package'].centers['RC'],saved[i]['center'],rtol=0,atol=1e-9)
        res=simulate(cfg,world,bundle,'RC',out=out/f'loop_{mission}',service_trackers=trackers)
        row={k:v for k,v in res.items() if not isinstance(v,(dict,list,np.ndarray))}
        row['tube']='native'
        results=pd.concat([results,pd.DataFrame([row])],ignore_index=True)
    results.to_csv(out/'closed_loop.csv',index=False)
    rows=[]
    for (method,tube),g in results.groupby(['method','tube']):
        assert len(g)==2 and np.allclose(g.accounted_s,g.service_s) and (g.sensing_fraction<=g.sensing_budget).all()
        seconds=(g.narrow_segments*g.narrow_hold_s.fillna(0)).sum()
        np.testing.assert_allclose(seconds,(g.narrow_share*g.service_s*g.N).sum())
        rows.append(dict(method=method,tube=tube,net_mbps=g.net_mbps.mean(),narrow_share=g.narrow_share.mean(),
                    narrow_hold_s=seconds/g.narrow_segments.sum() if g.narrow_segments.sum() else None,
                    observations=g.observations_service.mean(),failures=g.hold_failures.sum(),segments=g.hold_segments.sum(),
                    narrow_failures=g.narrow_failures.sum(),narrow_segments=g.narrow_segments.sum()))
    df=pd.DataFrame(rows)
    df.to_csv(out/'summary.csv',index=False)
    print(df.to_string(index=False))


if __name__=='__main__':
    main()
