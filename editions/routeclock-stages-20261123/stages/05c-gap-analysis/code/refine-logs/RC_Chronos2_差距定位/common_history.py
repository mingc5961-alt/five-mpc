"""Replay seed 7 at the final warm-up observation. Run with the published campaign environment."""
import argparse
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/data/chen/routeclock_runs/confirm_0db')
sys.path[:0]=[str(ROOT),str(ROOT/'work/code/routeclock_refine')]
from advanced_tubes import Predictor,get_bundle
from run_flights import MeasuredFlight,warm
from refine import network,world_cache
from routeclock.config import load_config
from routeclock.geometry import angles,angle_error
from routeclock.radio import Radio,array_gain
from runtime.remaining_models import CausalModel

def evaluate(shard,shards):
    cfg=load_config();cfg['simulation']['sensing_budget']=.05/3
    cache=ROOT/'work/baseline_cache';world_cache.BASE=cache
    classical=pickle.loads((cache/'classical.pkl').read_bytes())['RC']
    chronos=Predictor('Chronos-2',7);tube=get_bundle(chronos,7)
    flights=json.loads((ROOT/'work/flight_windows.json').read_text())
    ns=json.loads((ROOT/'scope.json').read_text())['analysis']['confirmation_namespace']
    jobs=[('合成',i) for i in range(40)]+[('实飞',i) for i in range(len(flights))]
    radio=Radio(cfg);radio.threshold-=5
    rows=[]
    for domain,i in jobs[shard::shards]:
        rc=CausalModel('RC',classical,cfg)
        if domain=='合成':
            unit=f'{i:03}';route='';w=world_cache.paired_world(cfg,ns,i);hs=network.warm_history(w,180,.0031)
            with gzip.open(cache/'warm'/f'v1_{i:03}.pkl.gz','rb') as f:trackers=pickle.load(f)
            for tr,history in zip(trackers,hs,strict=True):
                assert np.array_equal([o.t for o in tr.history],[o.t for o in history])
                np.testing.assert_allclose([o.position for o in tr.history],[o.position for o in history],atol=1e-9,rtol=0)
                history[:] = tr.history
                rc.trackers[id(history)]=tr
            paths=[ROOT/'results/baselines'/f'{unit}_T_RC/complete.json',ROOT/'results/confirmation/Chronos-2/7'/f'{unit}_T_native/complete.json']
        else:
            win=flights[i];unit=win['flight_id'];route=win['route'];w=MeasuredFlight(cfg,win,i);hs=warm(w,180,.0031)
            paths=[ROOT/'results/amovfly'/m/unit/'T/complete.json' for m in ('RC','Chronos-2')]
        models=[('RC',rc.batch(hs),classical['factors']['default']),('Chronos-2',chronos.batch(hs),tube['factors']['native'])]
        for (model,preds,factors),path in zip(models,paths,strict=True):
            digest=hashlib.sha256(np.stack([c for c,_ in preds]).tobytes()).hexdigest()
            match=digest==json.loads(path.read_text())['result']['initial_center_sha256']
            for uid,(center,scale) in enumerate(preds):
                origin=hs[uid][-1].t
                tt=np.arange(501)*.02
                truth=network.interpolate(w.positions[uid],origin+tt,w.dt)
                pred=np.column_stack([np.interp(tt,network.HS,v) for v in center.T])
                pos=np.linalg.norm(pred-truth,axis=1)
                angular=np.rad2deg(np.linalg.norm(angle_error(angles(pred),angles(truth)),axis=1))
                loss=10*np.log10(1024/array_gain(angles(pred),angles(truth),(32,32)))
                actions=network.frontier(radio,center,scale,factors)
                for h in (0,1,3,5,10):
                    ix=int(h*50);k=int(h*2);j=list(network.QUERY_H[1:]).index(h) if h else None
                    feasible=any(a.pair==(32,32) and a.interval==h for a in actions) if h else None
                    width=np.rad2deg(scale[k]*factors[j]) if h else (np.nan,np.nan)
                    rows.append(dict(domain=domain,unit=unit,route=route,user=uid,model=model,h=h,
                            hash_match=match,position_m=pos[ix],angle_deg=angular[ix],loss32_db=loss[ix],
                            safe32_prefix=bool(np.max(loss[:ix+1])<=3),cert32=feasible,
                            az_halfwidth_deg=width[0],el_halfwidth_deg=width[1]))
    pd.DataFrame(rows).to_csv(Path(__file__).parent/f'anchors_{shard}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('shard',type=int);p.add_argument('shards',type=int);args=p.parse_args()
    evaluate(args.shard,args.shards)
