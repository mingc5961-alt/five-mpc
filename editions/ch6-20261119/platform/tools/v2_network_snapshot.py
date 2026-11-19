"""One N=16 joint decision after 180 s of shared-budget warmup, using frozen pilot heads."""
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pilot_histories import ROOT,OUT
from routeclock.config import load_config,dump_json
from routeclock.world import World
from routeclock.predictors import Tracker,apply_gate
from routeclock.radio import Radio
from routeclock.scheduler import frontier,choose_actions


def prepare(observations):
    tracker=Tracker(load_config())
    for obs in observations:
        tracker.observe(obs)
        pkg=tracker.issue()
    return pkg


if __name__=='__main__':
    cfg=load_config()
    sim=cfg['simulation']
    cost,budget=sim['observation_cost_s'],sim['sensing_budget']
    world=World(cfg,'network/v2_diagnostic',0,n=16)
    # Stagger each aircraft by the token refill time; total load 16*c per second.
    times=np.arange(180)[:,None]+np.arange(16)*cost/budget+cost
    assert times.max()<180 and 16*cost<=budget
    obs=[[world.observe(uid,t) for t in times[:,uid]] for uid in range(16)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        packages=list(pool.map(prepare,obs))
    with (OUT/'results/bundle.pkl').open('rb') as stream:
        bundle=pickle.load(stream)
    fixed=json.loads((ROOT/'runs/fixed_validation/winner.json').read_text())
    radio=Radio(cfg)
    fronts=[]
    for pkg in packages:
        apply_gate(pkg,bundle['gates']['RC'],pkg.features,'RC','P')
        env=bundle['heads']['RC'].predict(pkg,pkg.features)
        fronts.append(frontier(pkg,'RC',env,radio,cfg))
    results={}
    for mode in ('joint','width_only','interval_only'):
        trace={}
        choices=choose_actions(fronts,cost,budget,width_only=mode=='width_only',interval_only=mode=='interval_only',
            fixed_interval=fixed['interval'],fixed_pair=tuple(fixed['pair']),trace=trace)
        trace['selected']=[a.__dict__ if a else None for a in choices]
        results[mode]=trace
    dump_json(ROOT/'runs/v2_review/n16_snapshot.json',results)
