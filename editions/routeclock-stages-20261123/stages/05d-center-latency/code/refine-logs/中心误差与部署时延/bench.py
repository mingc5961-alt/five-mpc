"""One model per process, fixed-rate 16-stream soft-real-time shadow replay."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np

OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'专用尺度管实验'
sys.path.insert(0,str(SOURCE))
from protocol import NS,PRIOR,ROOT
from replay import Advanced,ForecastTube,network,runtime,world_cache,load_config,Radio,frontier,reference
from advanced_tubes import Predictor,get_bundle
import torch

def bench(model,kind,n,w):
    cfg=load_config()
    cfg['simulation']['sensing_budget']=runtime.BUDGET['T']
    world_cache.BASE=ROOT/'work/baseline_cache'
    runtime.OUT=PRIOR/'cache'
    world=world_cache.paired_world(cfg,NS,w)
    cfg=load_config();rho=runtime.BUDGET['L'];cfg['simulation']['sensing_budget']=rho
    radio=Radio(cfg);radio.threshold-=5
    h=network.warm_history(world,180.,.0031)
    users=len(h)
    an=64;step=.0031/rho;clock=time.perf_counter_ns
    assert 180.+(n+an)*step < world.times[-1]
    # Synthetic sensor packets already in memory; generation is outside the measured path.
    packets=[world.observe(i%users,180.+(i+1)*step) for i in range(n+an)]
    cpu=model=='RC'
    sync=(lambda:None) if cpu else torch.cuda.synchronize
    load_start=clock(); fixed=None
    if kind=='special':
        head=pickle.loads((SOURCE/'heads'/model/'special.pkl').read_bytes())
        net=ForecastTube(model,None if cpu else Advanced(model),head,world)
        factors=head['factors']
    elif cpu:
        par=pickle.loads((PRIOR/'cache/classical.pkl').read_bytes())['RC']
        net=runtime.CausalModel('RC',par,cfg)
        for hist,tr in zip(h,runtime.warm_trackers(world),strict=True):
            assert hist[-1].t==tr.history[-1].t and len(hist)==len(tr.history)
            net.trackers[id(hist)]=tr
        factors=par['factors']['default']
    else:
        net=Predictor(model,7);par=get_bundle(net,7)
        native=model=='Chronos-2';factors=par['factors']['native' if native else 'fixed']
        if not native:fixed=par['fixed']
    sync()
    loaded=clock()
    starts=[x[-1].t for x in h]
    initial=net.batch(h);sync()
    fp=hashlib.sha256(np.stack([x for x,_ in initial]).tobytes()).hexdigest()
    assert fp==json.loads(reference(model,w,'L').read_text())['result']['initial_center_sha256']
    fronts=[frontier(radio,c,raw if fixed is None else fixed,factors) for c,raw in initial]
    def select(t,updated,old):
        candidates=[]
        for i,fs in enumerate(fronts):
            if updated is not None and i!=updated and (old[i] is None or starts[i]+old[i].interval<=t):
                candidates.append([])
            else:
                candidates.append([a for a in fs if starts[i]+a.interval>t])
        return network.choose_actions(candidates,.0031,rho)
    actions=select(180.,None,None)
    initialized=clock()
    rows=[]
    t0=clock()
    for i,obs in enumerate(packets):
        uid=i%users;due=t0+round((i+1)*step*1e9)
        wait=(due-clock())*1e-9
        if wait>0:time.sleep(wait)
        start=clock()
        h[uid].append(obs);starts[uid]=obs.t
        [(center,raw)]=net.batch([h[uid]]);sync()
        predicted=clock()
        fronts[uid]=frontier(radio,center,raw if fixed is None else fixed,factors)
        actions=select(obs.t,uid,actions)
        # Host-resident beam IDs/expiry available to an actuator adapter.
        cmd=tuple(None if a is None else (a.pair,starts[j]+a.interval) for j,a in enumerate(actions))
        end=clock()
        rows.append((i,uid,i>=an,(start-due)/1e6,(predicted-start)/1e6,(end-predicted)/1e6,(end-start)/1e6,(end-due)/1e6))
    base=OUT/f'{model}_{kind}'
    with base.with_suffix('.csv').open('w') as f:
        wt=csv.writer(f);wt.writerow(['event','uid','scored','wait_ms','forecast_head_ms','schedule_ms','service_ms','response_ms']);wt.writerows(rows)
    a=np.asarray(rows[an:]);v=a[:,-1]
    report=dict(model=model,head=kind,world=w,n=n,discard=an,period_ms=step*1000,users=users,
         device='cpu' if cpu else torch.cuda.get_device_name(),torch=torch.__version__,threads=torch.get_num_threads(),
         load_and_restore_ms=(loaded-load_start)/1e6,initial_batch_with_checks_ms=(initialized-loaded)/1e6,
         p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),maximum=float(v.max()),
         late_10=int((v>10).sum()),late_20=int((v>20).sum()),late_62=int((v>step*1000).sum()),
         compute_p99=float(np.percentile(a[:,-2],99)),queue_p99=float(np.percentile(a[:,3],99)),
         forecast_p50=float(np.median(a[:,4])),schedule_p50=float(np.median(a[:,5])))
    base.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('model',choices=['RC','Chronos-2','VECTOR','RAFT']);parser.add_argument('head',choices=['original','special'])
    parser.add_argument('--events',type=int,default=1024); parser.add_argument('--world',type=int,default=0)
    a=parser.parse_args();bench(a.model,a.head,a.events,a.world)
