"""Hold the old observation clocks and all 60 position forecasts fixed."""
import json
from types import SimpleNamespace
from chronos_tube_pilot import (
    ROOT, OUT, ChronosAdapter, QuantileTube, Radio, configuration, frontier,
    read_pickle, summarize, np, pd, torch,
)
from routeclock.radio import array_gain
from routeclock.geometry import angles


def main():
    torch.set_num_threads(2)
    old = OUT/'results'
    out = ROOT/'runs/chronos_tube_cpu'
    cfg = configuration()
    radio = Radio(cfg)
    manifest = json.loads((old/'manifest.json').read_text())
    model = ChronosAdapter('cpu', str((ROOT/manifest['checkpoint']).resolve()))
    head = QuantileTube()
    head.factor = json.loads((out/'calibration.json').read_text())['factor']
    rows = []
    for mission in range(2):
        _, trackers, world = read_pickle(OUT/f'network_{mission:02}.pkl')
        history = [list(tr.history) for tr in trackers]
        logs = json.loads((old/f'loop_{mission}_C_joint/predictions.json').read_text())
        items = []
        for log in logs:
            uid, origin = log['uav'], log['origin']
            if origin > history[uid][-1].t:
                history[uid].append(world.observe(uid, origin))
            assert origin == history[uid][-1].t
            inp, offset = model.inputs(history[uid], [], None, None, None, False)
            items.append((inp, offset, history[uid][-1].covariance))
        # Service forecasts were originally issued one at a time. Mixed-length
        # batching changes input padding, so reproduce the deployment call shape.
        batches = [slice(0,3)] + [slice(j,j+1) for j in range(3,len(logs))]
        for batch in batches:
            for log,(pos,cov) in zip(logs[batch],model.predict_inputs(items[batch]),strict=True):
                center = np.array(log['center'])
                mismatch = np.max(np.abs(pos-center))
                np.testing.assert_allclose(pos, center, atol=.001, rtol=0)
                pkg = SimpleNamespace(centers={'C':center},covariance={'C':cov},hs=np.array(log['h']))
                origin = log['origin']
                ts = np.arange(501)*.02
                truth = np.array([world.truth(log['uav'],origin+t)[0] for t in ts])
                pred = np.array([np.interp(ts,pkg.hs,axis) for axis in center.T]).T
                loss = -10*np.log10(array_gain(angles(pred),angles(truth),(32,32))/1024)
                for name,env in [('lightgbm',np.array(log['envelope_rad'])),('native',head.predict(pkg,None))]:
                    fs = frontier(pkg,'C',env,radio,cfg)
                    allowed = max((a.interval for a in fs if a.pair==(32,32)),default=0.)
                    rows.append(dict(mission=mission,uid=log['uav'],origin=origin,tube=name,
                        center_max_difference_m=mismatch,approved_s=allowed,
                        early_failure=bool(allowed and np.any(loss[ts<=allowed]>3))))
    data = pd.DataFrame(rows)
    assert len(data)==120
    data.to_csv(out/'fixed_center_replay.csv',index=False)
    if (out/'closed_loop.csv').exists():
        results = pd.read_csv(out/'closed_loop.csv')
        if len(results)==6:
            summarize(results,out)


if __name__=='__main__':
    main()
