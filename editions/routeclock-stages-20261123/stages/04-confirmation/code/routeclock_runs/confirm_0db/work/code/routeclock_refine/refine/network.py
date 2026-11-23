"""Paired model/tube closed loop, with expiry, public scoring grid and shared budget."""
import argparse
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from routeclock.config import load_config
from routeclock.geometry import angles
from routeclock.radio import Radio,array_gain
from routeclock.scheduler import Action,choose_actions,TokenBucket
from routeclock.world import World
from .models import HS
from .world_cache import paired_world
from .data import BASE
from .tubes import MODELS,features,tree_scale,QUERY_H

def interpolate(series,times,dt):
    ix=np.floor(times/dt).astype(int)
    frac=times/dt-ix
    if series.ndim==2:frac=frac[:,None]
    return series[ix]*(1-frac)+series[ix+1]*frac

def frontier(radio,center,scale,factors):
    horizons=QUERY_H[1:]
    mask=HS[None,:]<=horizons[:,None]
    direction=angles(center)
    # The candidate's calibration is shared by both axes and the entire prefix.
    # Take adjacent maxima to cover interpolation between each pair of forecast knots.
    envelope=np.maximum(scale,np.vstack([scale[:1],scale[:-1]]))
    bounds=(envelope[None,:,:]*factors[:,None,None]).reshape(-1,2)
    directions=np.tile(direction,(len(horizons),1))
    actions=[]
    for pair in radio.pairs:
        nominal=radio.snr(center,direction,pair)
        mcs=np.searchsorted(radio.threshold,nominal,side='right').clip(1,len(radio.se))-1
        losses=radio.worst_losses(directions,bounds,pair).reshape(len(horizons),len(HS))
        good,_=radio.goodput(nominal-np.minimum(losses,40),mcs)
        worst=np.where(mask,losses,0.).max(1)
        feasible=np.logical_and.accumulate(worst<=3.)
        rates=np.sum(good*mask,axis=1)/mask.sum(1)
        actions.extend(Action(pair,float(h),float(r),float(l)) for h,r,l in zip(horizons[feasible],rates[feasible],worst[feasible]))
    return actions

def warm_history(world, warm, cost):
    n=world.n
    histories=[[] for _ in range(n)]
    period=max(1.,n*cost/.05)
    for uid in range(n):
        for t in np.arange(cost+uid*cost/.05,warm+1e-9,period):
            histories[uid].append(world.observe(uid,float(t)))
    return histories

class Ledger:
    def __init__(self,cfg,world,radio,warm,duration):
        self.world,self.radio,self.warm,self.duration=world,radio,warm,duration
        self.probes=warm+(np.arange(round(duration/.02))+.5)*.02
        self.risk_count=np.zeros(world.n,int)
        self.last_probe=0
        self.unavailable_count=np.zeros(world.n,int)
        self.probe_count=0
        self.bits=np.zeros(world.n)
        self.times=np.zeros(4)  # sensing, switching, usable data, unavailable data
        self.rr=0
        self.quantum=cfg['simulation']['tdma_quantum_s']
        self.switch=cfg['simulation']['beam_switch_s']
        self.narrow=0.

    def advance(self,start,finish,state,sensing):
        if finish<=start:return
        end_probe=np.searchsorted(self.probes,finish,side='left')
        public=self.probes[self.last_probe:end_probe]
        self.probe_count+=len(public)
        self.last_probe=end_probe
        if sensing:
            self.times[0]+=finish-start
            users=np.empty(0,int);starts=ends=np.empty(0)
        else:
            count=max(1,int(np.ceil((finish-start)/self.quantum-1e-10)))
            slots=start+np.arange(count)*self.quantum
            ends=np.minimum(slots+self.quantum,finish)
            starts=np.minimum(slots+self.switch,ends)
            users=(self.rr+np.arange(count))%self.world.n
            self.rr=(self.rr+count)%self.world.n
            self.times[1]+=sum(starts-slots)
        for uid,p in enumerate(state):
            selected=users==uid
            left,right=starts[selected],ends[selected]
            available=(p.action is not None)
            expiry=p.origin+p.action.interval if available else start
            good=public<expiry if available else np.zeros(len(public),bool)
            self.unavailable_count[uid]+=(~good).sum()
            self.risk_count[uid]+=(~good).sum()
            stop=np.maximum(left,np.minimum(right,expiry))
            self.times[3]+=(right-stop).sum()
            used=stop>left
            if not available or not (good.any() or used.any()):continue
            dt=(stop-left)[used]
            mids=(stop[used]+left[used])*.5
            ts=np.r_[public[good],mids]
            pred=np.column_stack([np.interp(ts-p.origin,HS,v) for v in p.center.T])
            pointing=angles(pred)
            xyz=interpolate(self.world.positions[uid],ts,self.world.dt)
            pair=p.action.pair
            loss=10*np.log10(np.prod(pair)/array_gain(pointing,angles(xyz),pair))
            split=good.sum()
            self.risk_count[uid]+=(loss[:split]>3.).sum()
            if not len(mids):continue
            self.times[2]+=dt.sum()
            nominal=self.radio.snr(pred[split:],pointing[split:],pair)
            mcs=np.searchsorted(self.radio.threshold,nominal,side='right').clip(1,len(self.radio.se))-1
            shadow=interpolate(self.world.shadows[uid],mids,self.world.dt)
            actual=self.radio.snr(xyz[split:],pointing[split:],pair,shadow)
            rate,_=self.radio.goodput(actual,mcs)
            self.bits[uid]+=rate@dt
            if pair==(32,32):self.narrow+=dt[loss[split:]<=3.].sum()

    def result(self,cost,rho):
        assert abs(sum(self.times)-self.duration)<1e-6, self.times
        assert self.times[0]<=rho*self.duration+cost+1e-8
        assert self.probe_count==len(self.probes)
        return dict(R=float(self.risk_count.mean()/len(self.probes)),G_mbps=float(self.bits.sum()/self.duration/1e6),
                    unavailable_fraction=float(self.unavailable_count.mean()/len(self.probes)),
                    ledger_s=self.times.tolist(),bits=self.bits.tolist(),probes_per_user=self.probe_count,narrow_good_data_s=self.narrow)

def run_world(cfg,world,model,bundle,tube,trace=False):
    sim=cfg['simulation'];warm=sim['warmup_s'];duration=sim['service_s'];end=warm+duration
    cost=.0031;rho=sim['sensing_budget'];bucket=TokenBucket(cost,rho);radio=Radio(cfg)
    histories=warm_history(world,warm,cost)
    state=[SimpleNamespace(action=None) for _ in histories]
    ledger=Ledger(cfg,world,radio,warm,duration)
    events=[];changes=0;selection=np.zeros(len(QUERY_H)-1,int)
    def publish(i,pred):
        h=histories[i];center,scale=pred
        if tube=='lightgbm':scale=tree_scale(bundle['tree'],features(h,center)[None])[0]
        if tube=='fixed':scale=bundle['fixed']
        p=state[i];p.center=center;p.origin=h[-1].t
        p.front=frontier(radio,center,scale,bundle['factors'][tube])
    def choose(t,updated=None):
        fronts=[]
        for i,p in enumerate(state):
            if updated is not None and i!=updated and (p.action is None or p.origin+p.action.interval<=t):
                fronts.append([])
            else:
                fronts.append([a for a in p.front if p.origin+a.interval>t])
        for p,a in zip(state,choose_actions(fronts,cost,rho),strict=True):p.action=a
        if trace:
            events.append(dict(t=t,updated=updated,states=[dict(origin=p.origin,center=p.center.tolist(),
                  pair=list(p.action.pair) if p.action else None,deadline=p.action.interval if p.action else None) for p in state]))
    initial=model.batch(histories)
    fingerprint=hashlib.sha256(np.stack([c for c,_ in initial]).tobytes()).hexdigest()
    for uid,pred in enumerate(initial):publish(uid,pred)
    choose(warm)
    now=warm
    while now<end-1e-10:
        due=[p.origin+(p.action.interval-cost if p.action else 0.) for p in state]
        uid=int(np.argmin(due))
        start=min(end,max(now,due[uid],bucket.ready(now)))
        ledger.advance(now,start,state,False)
        if start+cost>end:
            ledger.advance(start,end,state,False)
            break
        bucket.consume(start)
        finish=start+cost
        ledger.advance(start,finish,state,True)
        histories[uid].append(world.observe(uid,finish))
        publish(uid,model.batch([histories[uid]])[0])
        choose(finish,uid)
        if state[uid].action:
            selection[np.searchsorted(QUERY_H[1:],state[uid].action.interval)]+=1
        changes+=1;now=finish
    result=ledger.result(cost,rho)
    result.update(initial_center_sha256=fingerprint,updates=changes,selected_deadlines=selection.tolist())
    return result,events

def execute(name,phase,ids,tubes,out,short=False):
    with (BASE/('pilot' if short else 'weights')/(name+'_tubes.pkl')).open('rb') as f:bundle=pickle.load(f)
    model=MODELS[name]();cfg=load_config();cfg['simulation']['sensing_budget']=.05/3
    if short: cfg['simulation'].update(primary_N=4,warmup_s=8.,service_s=6.)
    for idx in ids:
        target=out/phase/name/f'{idx:03}';target.mkdir(parents=True,exist_ok=True)
        world=None
        for tube in tubes:
            file=target/(tube+'.json')
            if file.exists():continue
            if world is None:world=paired_world(cfg,'refine_v1/'+phase,idx)
            tic=time.monotonic()
            result,events=run_world(cfg,world,model,bundle,tube,trace=short)
            result.update(model=name,tube=tube,phase=phase,world=idx,wall_s=time.monotonic()-tic,config=cfg['simulation'])
            if events:
                with gzip.open(target/(tube+'_events.json.gz'),'wt') as f:json.dump(events,f)
            tmp=file.with_suffix('.part');tmp.write_text(json.dumps(result,allow_nan=False));tmp.replace(file)
            print(name,phase,idx,tube,result['wall_s'],flush=True)
        pair=[json.loads((target/(t+'.json')).read_text()) for t in tubes]
        if len({r['initial_center_sha256'] for r in pair})!=1:raise ValueError('initial centers differ')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('name',choices=MODELS);ap.add_argument('--phase',choices=['dev','validation','confirmation'],required=True)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=1);ap.add_argument('--short',action='store_true')
    a=ap.parse_args()
    if a.short and a.phase!='dev':ap.error('short worlds are development-only')
    execute(a.name,a.phase,range(a.start,a.start+a.count),['native','lightgbm'],BASE/'tube_network',a.short)
