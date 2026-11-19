"""Score all codebook widths on each identical fixed-interval event trace."""
import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from routeclock.config import load_config, dump_json
from routeclock.filters import PositionIMM, position_transitions
from routeclock.scheduler import TokenBucket
from routeclock.geometry import angles
from routeclock.radio import Radio
from routeclock.world import World


def forecast_operator():
    # Joint first moments are linear during an unobserved IMM forecast.
    transition, matrices=position_transitions(.5)
    op=np.eye(18).reshape(3,6,18)
    positions=[]
    for _ in range(21):
        positions.append(op.sum(axis=0)[:3])
        op=np.einsum('jab,ij,ibk->jak',matrices,transition,op)
    return np.array(positions).reshape(63,18)


def fixed_scores(cfg,world,service_trackers=None):
    sim=cfg['simulation']; n=world.n; cost=sim['observation_cost_s']
    budget=sim['sensing_budget']; warm=sim['warmup_s']; end=warm+sim['service_s']
    warm_step=max(1.,n*cost/budget)
    radio=Radio(cfg); op=forecast_operator(); rows=[]
    # The state immediately before the first service event is shared by all intervals.
    shared=None
    for interval in sim['intervals_s']:
        if n*cost/interval>budget:continue
        if service_trackers is not None:
            filters=copy.deepcopy([tr.imm for tr in service_trackers])
            origins=np.array([tr.snapshots[-1].origin for tr in service_trackers])
            centers=np.array([tr.snapshots[-1].centers['I'] for tr in service_trackers])
            t=warm; rr=0; due=origins+interval; bucket=TokenBucket(cost,budget)
        elif shared is None:
            t=0.; rr=0; due=np.arange(n)*cost/budget
            bucket=TokenBucket(cost,budget); filters=[PositionIMM() for _ in range(n)]
            origins=np.full(n,-np.inf); centers=np.zeros((n,21,3))
        else:
            t,rr,due,bucket,filters,origins,centers=copy.deepcopy(shared)
        durations=[]; positions=[]; uids=[]; midtimes=[]
        while t<end-1e-10:
            if service_trackers is None and shared is None and t+max(cost,sim['tdma_quantum_s'])>=warm:
                shared=copy.deepcopy((t,rr,due,bucket,filters,origins,centers))
            i=int(np.argmin(due)); earliest=max(float(due[i]),bucket.ready(t))
            if earliest<=t+1e-9 and min(t+cost,end)-t>=cost-1e-10:
                bucket.consume(t);t=min(t+cost,end)
                filt=filters[i]; filt.observe(world.observe(i,t));origins[i]=t
                if t>=warm-warm_step-1:
                    centers[i]=(op@(filt.weights[:,None]*filt.x).ravel()).reshape(21,3)
                known=np.isfinite(origins)
                due[known]=origins[known]+(warm_step if t<warm else interval)
                continue
            if earliest<=t+1e-9:
                finish=end;switch=0.
            else:
                finish=min(t+sim['tdma_quantum_s'],earliest,end)
                switch=min(sim['beam_switch_s'],finish-t)
            left=max(warm,t+switch); mid=(left+finish)/2;uid=rr%n
            age=mid-origins[uid]
            if finish>left and age<=10+1e-8:
                step=np.clip(age*2,0,20);j=int(step)
                pos=centers[uid,j]+(step-j)*(centers[uid,min(j+1,20)]-centers[uid,j])
                positions.append(pos);uids.append(uid);midtimes.append(mid);durations.append(finish-left)
            rr+=1;t=finish
        dur=np.array(durations); pos=np.asarray(positions);uid=np.asarray(uids)
        mid=np.asarray(midtimes);j=(mid/world.dt).astype(int);u=(mid-world.times[j])/world.dt
        actual=world.positions[uid,j]*(1-u[:,None])+world.positions[uid,j+1]*u[:,None]
        sh=np.array(world.shadows);shadow=sh[uid,j]*(1-u)+sh[uid,j+1]*u
        pointing=angles(pos)
        for pair in radio.pairs:
            expected=radio.snr(pos,pointing,pair)
            mcs=np.clip(np.searchsorted(radio.threshold,expected,side='right')-1,0,len(radio.se)-1)
            rate,_=radio.goodput(radio.snr(actual,pointing,pair,shadow),mcs)
            rows.append(dict(pair=pair,interval=interval,net_mbps=float(dur@rate/sim['service_s']/1e6)))
    return rows


def run_one(job):
    mission,directory=job;cfg=load_config()
    path=directory/f'mission_{mission:02d}.json'
    if not path.exists():
        rows=fixed_scores(cfg,World(cfg,'network/validation',mission))
        temp=path.with_suffix('.tmp');dump_json(temp,rows);temp.replace(path)
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--out',type=Path,default=ROOT/'runs/fixed_validation')
    parser.add_argument('--verify',action='store_true',help='Also replay a full-length candidate through the standard event simulator')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scenarios=[]
        for i,rows in enumerate(pool.map(run_one,[(i,args.out) for i in range(20)])):
            scenarios.append(rows);print('completed',i+1,'/20',flush=True)
    candidates=[]
    for group in zip(*scenarios):
        candidates.append(dict(pair=group[0]['pair'],interval=group[0]['interval'],
                               validation_mbps=float(np.mean([r['net_mbps'] for r in group]))))
    candidates.sort(key=lambda r:(r['pair'],r['interval']))
    dump_json(args.out/'candidates.json',candidates)
    dump_json(args.out/'winner.json',max(candidates,key=lambda r:r['validation_mbps']))
    if args.verify:
        from routeclock.simulator import simulate
        t=time.perf_counter();cfg=load_config()
        row=next(r for r in scenarios[0] if r['interval']==2 and r['pair']==[16,32])
        reference=simulate(cfg,World(cfg,'network/validation',0),dict(kind='fixed',seed=7,fixed=row),'F')
        error=abs(reference['net_mbps']-row['net_mbps'])
        if error>=1e-6:raise AssertionError(f'Fixed policy differs by {error} Mbps')
        dump_json(args.out/'full_reference_check.json',dict(mission=0,candidate=row,event_simulator_mbps=reference['net_mbps'],
                      absolute_difference_mbps=error,elapsed_s=time.perf_counter()-t))
if __name__=='__main__':main()

