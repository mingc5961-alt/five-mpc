"""Execute frozen-config plan cells with the shared scoring clock and ledger."""
from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import json
import pickle
import time
from types import SimpleNamespace
import numpy as np
from refine import network
from refine.data import BASE
from routeclock.config import load_config
from routeclock.evidence import Forecast
from routeclock.filters import PositionIMM
from routeclock.predictors import Tracker
from routeclock.radio import Radio
from routeclock.scheduler import Action,TokenBucket
from runtime.fast_ledger import DeferredLedger
from runtime.remaining_models import OUT,HS,CausalModel,RouteQuantiles
from runtime.learned_queue import digest,save

BUDGET={'T':.05/3,'M':.025,'L':.05}
def fixed_choice():
    with (BASE/'q1_frontier/F_scores.csv').open() as f:rows=list(csv.DictReader(f))
    grid={}
    for r in rows:
        k=int(r['width_x']),int(r['width_y']),float(r['period_s'])
        grid.setdefault(k,[]).append(r)
    assert len(grid)==36 and all({int(r['world']) for r in vals}==set(range(10)) for vals in grid.values())
    scores=[(np.mean([float(r['R']) for r in v]),np.mean([float(r['G_mbps']) for r in v]),k) for k,v in grid.items()]
    eligible=sorted(((-g,r,k) for r,g,k in scores if .019<=r<=.021))
    if not eligible:raise ValueError('F has no eligible operating point')
    neg_g,r,(x,y,period)=eligible[0]
    return dict(pair=[x,y],period=period,search_R=r,search_G_mbps=-neg_g,grid_sha256=digest(BASE/'q1_frontier/F_scores.csv'))

def fixed_run(cfg,world,radio,selection):
    warm=cfg['simulation']['warmup_s'];duration=cfg['simulation']['service_s'];rho=cfg['simulation']['sensing_budget']
    cost=.0031;end=warm+duration;period=selection['period']
    action=Action(tuple(selection['pair']),period,0.,0.)
    histories=network.warm_history(world,warm,cost);filters=[];states=[]
    for hist in histories:
        imm=PositionIMM()
        for obs in hist:imm.observe(obs)
        states.append(SimpleNamespace(origin=hist[-1].t,action=action,center=imm.forecast(HS)[0]))
        filters.append(imm)
    center_hash=hashlib.sha256(np.stack([s.center for s in states]).tobytes()).hexdigest()
    ledger=DeferredLedger(cfg,world,radio,warm,duration);bucket=TokenBucket(cost,rho)
    due=np.array([s.origin+period-cost for s in states]);now=warm;updates=0
    while now<end-1e-10:
        uid=int(np.argmin(due));at=max(now,due[uid],bucket.ready(now))
        if at+cost>end:
            ledger.advance(now,end,states,False);break
        ledger.advance(now,at,states,False)
        bucket.consume(at);now=at+cost;ledger.advance(at,now,states,True)
        filters[uid].observe(world.observe(uid,now))
        states[uid].center=filters[uid].forecast(HS)[0];states[uid].origin=now
        due[uid]=now+period-cost;updates+=1
    res=ledger.result(cost,rho);res.update(initial_center_sha256=center_hash,updates=updates,fixed=selection)
    return res

def warm_trackers(world):
    folder=OUT/'warm';folder.mkdir(exist_ok=True)
    key=f'{world.namespace.split("/")[-1]}_{world.mission:03}'
    path=folder/(key+'.pkl.gz')
    with (folder/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            with gzip.open(path,'rb') as f:return pickle.load(f)
        trackers=[]
        for hist in network.warm_history(world,world.cfg['simulation']['warmup_s'],.0031):
            tr=Tracker(world.cfg)
            for obs in hist:tr.observe(obs);tr.issue()
            # Preserve the complete issued route information consumed by Chronos and all live evidence.
            for j,p in enumerate(tr.snapshots[:-1]):
                tr.snapshots[j]=Forecast(p.origin,p.hs,{'P':p.centers['P']},{'P':p.covariance['P']},None,{'support':p.diagnostics['support']})
            trackers.append(tr)
        tmp=path.with_suffix('.part')
        with gzip.open(tmp,'wb',compresslevel=1) as f:pickle.dump(trackers,f)
        tmp.replace(path)
        return trackers

@contextmanager
def controller(radio,max_interval):
    old_radio,old_front,old_ledger=network.Radio,network.frontier,network.Ledger
    def frontier(*a):
        candidates=old_front(*a)
        return [c for c in candidates if c.interval<=max_interval]
    network.Radio=lambda cfg:radio
    network.frontier=frontier;network.Ledger=DeferredLedger
    try: yield
    finally:network.Radio,network.frontier,network.Ledger=old_radio,old_front,old_ledger

class Runner:
    def __init__(self):
        self.chronos=None;self.bundles={}
    def execute(self,row):
        name=row['model'];cfg=load_config();cfg['simulation']['sensing_budget']=BUDGET['T']
        world=network.paired_world(cfg,'refine_v1/'+row['phase'],row['world'])
        # Only the simulator's budget and radio margin vary. All arms reuse the same physical arrays.
        cfg=load_config();cfg['simulation']['sensing_budget']=BUDGET[row['condition']]
        radio=Radio(cfg);radio.threshold+=row['mcs_margin_db']-5
        if name=='F':
            sel=json.loads((OUT/'F.json').read_text())
            res=fixed_run(cfg,world,radio,sel)
        else:
            fn=name if name in ('CR-G','CR_raw') else 'classical'
            if fn not in self.bundles:self.bundles[fn]=pickle.loads((OUT/(fn+'.pkl')).read_bytes())
            params=self.bundles[fn] if fn==name else self.bundles[fn][name]
            if name in ('CR-G','CR_raw') and self.chronos is None:self.chronos=RouteQuantiles()
            model=CausalModel(name,params,cfg,self.chronos)
            trackers=warm_trackers(world)
            initial=model.batch
            def init(histories):
                assert len(histories)==len(trackers)
                for h,tr in zip(histories,trackers,strict=True):
                    assert len(tr.history)==len(h) and h[-1].t==tr.history[-1].t
                    model.trackers[id(h)]=tr
                model.batch=initial
                return initial(histories)
            model.batch=init
            tube=row['tube']
            method='fixed' if tube=='fixed' else 'default'
            par=dict(params,factors={method:params['factors'][method]*row.get('risk_scale',1.)})
            with controller(radio,row.get('max_interval',10.)):
                res,_=network.run_world(cfg,world,model,par,method)
            res.update(bundle_sha256=digest(OUT/(fn+'.pkl')),risk_scale=row.get('risk_scale',1.),max_interval=row.get('max_interval',10.))
        res.update(row,config=cfg['simulation'])
        assert res['probes_per_user']==9000 and len(res['bits'])==16
        assert abs(sum(res['ledger_s'])-180)<1e-6 and abs(sum(res['bits'])/180/1e6-res['G_mbps'])<1e-7
        return res
