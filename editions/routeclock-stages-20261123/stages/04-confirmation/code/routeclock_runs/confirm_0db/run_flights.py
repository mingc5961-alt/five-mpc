"""Paired single-aircraft C3 replay with frozen synthetic models."""
from contextlib import ExitStack
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import pickle
import time
from unittest.mock import patch
import numpy as np
import pandas as pd

from freeze_flights import horizontal, ROOT, WORK
from refine import network
from routeclock.config import load_config
from routeclock.radio import Radio
from routeclock.world import World, rng_for
from runtime.remaining_models import CausalModel
from runtime.remaining_runs import BUDGET, controller

def sha(b):
    return hashlib.sha256(b).hexdigest()

class MeasuredFlight(World):
    def __init__(self, cfg, win, index):
        raw=(ROOT/win['source']).read_bytes()
        assert sha(raw)==win['source_sha256']
        d=pd.read_csv(io.BytesIO(raw))
        start=win['first_t']
        # Keep a bracketing sample beyond the last truth grid point.
        lo=int(np.flatnonzero(np.isclose(d.time,start,rtol=0,atol=1e-7))[0])
        t=d.time.to_numpy()[lo:]-start
        hi=int(np.searchsorted(t,370.02-1e-8))+1
        t=t[:hi];d=d.iloc[lo:lo+hi]
        assert t[-1]>=370.02-1e-8 and (np.diff(t)>0).all() and np.diff(t).max()<=.25
        self.dt=.02;self.times=np.arange(18502)*self.dt
        self.n=1;self.cfg=cfg;self.namespace='amovfly_c3_0db_v1';self.mission=index
        xy=horizontal(d.real_lat,d.real_long,win['site_ref'])
        enu=np.column_stack([xy,d.gps_z-win['ground_z']])-win['station_enu']
        self.positions=np.column_stack([np.interp(self.times,t,v) for v in enu.T])[None]
        # Measured MAVROS local velocity components are contemporaneous.
        vel=d[['v_x','v_y','v_z']].to_numpy()
        assert np.isfinite(vel).all()
        self.velocities=np.column_stack([np.interp(self.times,t,v) for v in vel.T])[None]
        rng=rng_for(self.namespace,self.mission)
        self.noise=rng.normal(size=(1,len(self.times),4))
        r=cfg['radio'];phi=np.exp(-self.dt/r['shadow_correlation_s'])
        shadow=rng.normal(0,r['shadow_std_db'],len(self.times))
        for i in range(1,len(shadow)):
            shadow[i]=phi*shadow[i-1]+np.sqrt(1-phi**2)*shadow[i]
        self.shadows=shadow[None]

def warm(world, seconds, cost):
    period=max(1.,cost/(.05/16))
    return [[world.observe(0,float(t)) for t in np.arange(cost,seconds+1e-9,period)]]

def quantile_scale(tree,features):
    from advanced_tubes import EPS
    return np.maximum(EPS,tree.predict(features.reshape(-1,features.shape[-1]),num_threads=1).reshape(-1,21,2))

def main(name,limit):
    manifest=(WORK/'flight_windows.json').read_bytes()
    wins=json.loads(manifest)
    classical=name in ('I','RC')
    model_file=WORK/'baseline_cache/classical.pkl' if classical else ROOT/'training'/name/'7/best.pt'
    cfg=load_config();cfg['simulation']['primary_N']=1
    contract=dict(manifest_sha256=sha(manifest),implementation_sha256=sha(Path(__file__).read_bytes()),
                  geometry_sha256=sha((ROOT/'freeze_flights.py').read_bytes()),
                  model_sha256=sha(model_file.read_bytes()),config_sha256=sha(json.dumps(cfg,sort_keys=True).encode()),
                  seed=7,model=name,namespace='amovfly_c3_0db_v1',mcs_margin_db=0,risk_scale=1,max_interval=10)
    if classical:
        bundle=pickle.loads(model_file.read_bytes())[name];tube='default'
    else:
        from advanced_tubes import Predictor,get_bundle
        net=Predictor(name,7);bundle=get_bundle(net,7)
        contract['bundle_sha256']=sha((ROOT/'tubes'/name/'7/bundle.pkl').read_bytes())
        tube=json.loads((ROOT/'tubes'/name/'selection.json').read_text())['tube']
    with ExitStack() as ctx:
        ctx.enter_context(patch.object(network,'warm_history',warm))
        if not classical:ctx.enter_context(patch.object(network,'tree_scale',quantile_scale))
        for idx,win in enumerate(wins[:limit]):
            world=MeasuredFlight(cfg,win,idx)
            packet=b''.join(np.asarray([o.t,*o.raw],dtype='<f8').tobytes() for o in warm(world,180,.0031)[0])
            for budget,rho in BUDGET.items():
                row=dict(**contract,flight_id=win['flight_id'],budget=budget,tube=tube,simulated_radio=True,
                         prewarm_sha256=sha(packet),flight_sha256=win['source_sha256'])
                folder=ROOT/'results/amovfly'/name/win['flight_id']/budget
                folder.mkdir(parents=True,exist_ok=True);out=folder/'complete.json'
                with (folder/'run.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    if out.exists():
                        assert json.loads(out.read_text())['contract']==row
                        continue
                    cfg['simulation']['sensing_budget']=rho/16
                    radio=Radio(cfg);radio.threshold-=5
                    center=CausalModel(name,bundle,cfg) if classical else net
                    start=time.monotonic()
                    with controller(radio,10.):
                        result,_=network.run_world(cfg,world,center,bundle,tube)
                    assert result['probes_per_user']==9000 and len(result['bits'])==1
                    assert abs(sum(result['ledger_s'])-180)<1e-6 and result['ledger_s'][0]<=180*rho/16+.00310001
                    assert abs(sum(result['bits'])/180/1e6-result['G_mbps'])<1e-7
                    tmp=out.with_suffix('.tmp')
                    tmp.write_text(json.dumps(dict(contract=row,result=result,elapsed_s=time.monotonic()-start),allow_nan=False))
                    tmp.replace(out)
                    print(name,win['flight_id'],budget,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('model',choices=['I','RC','RAFT','VECTOR','Chronos-2'])
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();main(args.model,args.limit)
