"""Score frozen forecasts at common absolute target times without taking service observations."""
import concurrent.futures
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from ch6_common_history import ROOT,MAIN
from ch6_experiment import prewarm
from ch6_cache import prepare
from ch6_run import write
from routeclock.config import load_config
from routeclock.geometry import angles,angle_error
from routeclock.predictors import apply_gate

def score(world_index):
    output=ROOT/'boundary'/f'{world_index:02}.json'
    if output.exists():return
    with (MAIN/'models/frozen/bundle.pkl').open('rb') as stream: model=pickle.load(stream)
    records=[]
    for cost,budget in [(3.1,5.),(6.2,5.),(9.3,5.),(12.4,5.),(3.1,2.5),(3.1,5/3)]:
        cfg=load_config()
        cfg['simulation'].update(observation_cost_s=cost/1000,sensing_budget=budget/100)
        w,trackers,_=prepare(cfg,'chapter6/test',world_index,cost/1000,budget/100,None,False,prewarm)
        fingerprint=hashlib.sha256()
        for uid,tr in enumerate(trackers):
            fingerprint.update(np.asarray([len(tr.history)],dtype='<i4').tobytes())
            for o in tr.history:
                for x in (o.t,o.raw,o.position,o.covariance,o.radial_velocity,o.radial_std):
                    fingerprint.update(np.asarray(x,dtype='<f8').tobytes())
            pkg=tr.snapshots[-1]
            f=tr.features(pkg,'P')
            apply_gate(pkg,model['gates']['RC'],f,'RC','P')
            for dt in (1.,3.,5.):
                target=180+dt
                age=target-pkg.origin
                assert 0<=age<=pkg.hs[-1]
                truth,_=w.truth(uid,target)
                err={}
                for m in ('I','P','RC'):
                    residual=np.rad2deg(angle_error(angles(pkg.at(m,age)),angles(truth)))
                    err[m]=float(np.mean(residual**2))
                records.append(dict(world=world_index,uid=uid,cost_ms=cost,budget_pct=budget,
                   horizon_after_start=dt,forecast_age_s=age,observations=len(tr.history),origin=pkg.origin,
                   mse_deg2=err))
        for r in records[-len(trackers)*3:]:r['history_sha256']=fingerprint.hexdigest()
    write(output,records)

if __name__=='__main__':
    (ROOT/'boundary').mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(score,range(30)):pass
