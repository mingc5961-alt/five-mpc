"""Matched synthetic closed-loop and common-history contrasts for the November 19/23 comparison."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
from unittest.mock import patch

import numpy as np
import torch
from chronos import Chronos2Pipeline

OUT = Path(__file__).resolve().parent
ROOT = Path('/data/chen/routeclock_runs/confirm_0db')
sys.path[:0] = [str(ROOT), str(ROOT/'work/code/routeclock_refine')]
from advanced_tubes import Predictor, fit, get_bundle
from train_chronos_long import CHECKPOINT
from refine import network, world_cache, tubes
from refine.models import Chronos, forecast
from refine.samples import paths, records
from routeclock.config import load_config
from routeclock.radio import Radio
from runtime import remaining_runs as runtime
from runtime.fast_ledger import DeferredLedger
from runtime.remaining_models import CausalModel
from parallel_frontier import frontier
from routeclock.geometry import angles, angle_error

CACHE = ROOT/'work/baseline_cache'
NAMESPACE='confirm_0db_20261122/v1'
WORLD_IDS = (0,10,20,30)
def write(path, obj):
    tmp=path.with_suffix('.part')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

class Legacy(Chronos):
    def __init__(self, length=720, cov=True):
        super().__init__()
        self.length=length
        self.cov=cov
        self.name='archived adapter'
        self.sha='legacy-'+str(length)+'-'+str(cov)+'-'+CHECKPOINT.resolve().name
    @torch.no_grad()
    def raw(self,histories):
        inputs=[self.adapter.inputs(h,[],None,None,None,False,max_steps=self.length) for h in histories]
        if not self.cov:
            for x,_ in inputs:x.pop('past_covariates')
        qs,_=self.adapter.pipeline.predict_quantiles([v[0] for v in inputs],prediction_length=20,quantile_levels=[.1,.5,.9],cross_learning=False,context_length=self.length)
        return [(*forecast(q.cpu().numpy().transpose(1,0,2)+off[None,:,None],h[-1]), None) for q,(_,off),h in zip(qs,inputs,histories,strict=True)]
    def batch(self,h):
        return [(p,w) for p,w,_ in self.raw(h)]

def predictor(name):
    if name.startswith('old'):
        return Legacy(360 if '360' in name else 720, 'nopast' not in name)
    net=Predictor('Chronos-2',7)
    if name=='frozen':
        del net.net
        net.net=Chronos2Pipeline.from_pretrained(str(CHECKPOINT),device_map='cuda',dtype=torch.float32).model.eval()
        net.sha='official-'+CHECKPOINT.resolve().name
    else: assert name=='tuned'
    return net

def parameters(name,net):
    if name=='tuned':return get_bundle(net,7)
    path=OUT/name/'bundle.pkl'
    if not path.is_file():return fit(net,7,path.parent)
    bundle=pickle.loads(path.read_bytes())
    assert bundle['center_sha256']==net.sha
    return bundle

def step_calibration(name):
    parameters(name,predictor(name))

if __name__=='__main__' and sys.argv[1]=='calibrate':step_calibration(sys.argv[2])

class ExpiryLedger(DeferredLedger):
    """Price the exact issued decisions twice. A missing action stays unavailable."""
    def result(self,cost,rho):
        import copy
        later=copy.copy(self)
        later.states=[[(origin,index,10. if interval>0 else 0.,nx,ny)
                       for origin,index,interval,nx,ny in snapshot] for snapshot in self.states]
        for key in ('unavailable_count','risk_count','times','bits'):
            setattr(later,key,getattr(self,key).copy())
        relaxed=DeferredLedger.result(later,cost,rho)
        strict=super().result(cost,rho)
        strict.update(ten_second_G=relaxed['G_mbps'],ten_second_risk=relaxed['R'])
        return strict

def sparse_history(world,warm,cost):
    rho=runtime.BUDGET['T']
    step=max(1.,world.n*cost/rho)
    return [[world.observe(uid,float(t)) for t in np.arange(cost+uid*cost/rho,warm+1e-9,step)] for uid in range(world.n)]

def run_job(name, idx, spec):
    """Change one factor from a common protocol and keep the physical draws fixed."""
    import contextlib
    scene,tube,warm,policy,margin=spec
    world_cache.BASE=runtime.OUT=CACHE
    cfg=load_config();cfg['simulation']['sensing_budget']=runtime.BUDGET['T']
    ns=NAMESPACE if scene=='new' else 'chapter6/test'
    w=(WORLD_IDS if scene=='new' else (0,2,5,7))[idx]
    world=world_cache.paired_world(cfg,ns,w)
    destination=OUT/'cells'/f'{scene}_{w:03}_{name}_{tube}_{warm}_{policy}_{margin}.json'
    if destination.is_file():return
    if name in ('RC','P','I','C19'):
        if name=='C19':
            from runtime.remaining_models import ARCHIVED, INDEX
            archive=pickle.loads(ARCHIVED.read_bytes())
            head=archive['heads']['C']
            bundle={'head':head,'factors':{'default':np.array([head.deadline_factors[float(h)] for h in network.QUERY_H[1:]])}}
            tracker_model=CausalModel('P',bundle,cfg)
            legacy=Legacy()
            class CenterWithArchivedHead:
                def __init__(self):self.trackers=tracker_model.trackers
                def batch(self,h):
                    estimates=legacy.batch(h)
                    heads=tracker_model.batch(h)
                    return [(c,width) for (c,_),(_,width) in zip(estimates,heads,strict=True)]
            net=CenterWithArchivedHead()
        else:
            bundle=pickle.loads((CACHE/'classical.pkl').read_bytes())[name]
            if tube=='same_RC':
                assert name in ('I','P')
                rcpar=pickle.loads((CACHE/'classical.pkl').read_bytes())['RC']
                bundle=dict(rcpar,factors={'same_RC':rcpar['factors']['default']})
            net=CausalModel(name,bundle,cfg)
            if name=='RC' and tube=='lightgbm':
                ext=pickle.loads((OUT/'RC_matched/bundle.pkl').read_bytes())
                bundle=dict(bundle,tree=ext['tree'],factors=dict(bundle['factors'],lightgbm=ext['factors']))
        if scene=='new' and warm=='dense':
            trackers=runtime.warm_trackers(world)
            original=net.batch
            def first(histories):
                for h,tr in zip(histories,trackers,strict=True):
                    np.testing.assert_allclose([o.position for o in h],[o.position for o in tr.history],rtol=0,atol=1e-7)
                    net.trackers[id(h)]=tr
                net.batch=original
                return original(histories)
            net.batch=first
    else:
        net=predictor(name);bundle=parameters(name,net)
    radio=Radio(cfg);radio.threshold+=margin-5
    with contextlib.ExitStack() as ctx:
        ctx.enter_context(patch.object(network,'Radio',lambda config:radio))
        ctx.enter_context(patch.object(network,'Ledger',ExpiryLedger))
        if name not in ('RC','P','I'):
            ctx.enter_context(patch.object(network,'frontier',frontier))
        if policy=='conservative':
            import inspect
            from parallel_frontier import frontier as base
            source=inspect.getsource(base)
            select="mcs=np.searchsorted(radio.threshold,nominal,side='right').clip(1,len(radio.se))-1"
            price="good,_=radio.goodput(nominal[:,None,:]-np.minimum(losses,40),mcs[:,None,:])"
            assert select in source and price in source
            source=source.replace(select,'').replace(price,"snr=nominal[:,None,:]-np.minimum(losses,40)\n    mcs=np.searchsorted(radio.threshold,snr,side='right').clip(1,len(radio.se))-1\n    good,_=radio.goodput(snr,mcs)")
            local={}
            exec(source,base.__globals__,local)
            ctx.enter_context(patch.object(network,'frontier',local['frontier']))
        if tube=='lightgbm':
            def qscale(tree,features):
                return np.maximum(tubes.EPS,tree.predict(features.reshape(-1,features.shape[-1]),num_threads=1).reshape(-1,21,2))
            ctx.enter_context(patch.object(network,'tree_scale',qscale))
        if warm=='sparse':
            ctx.enter_context(patch.object(network,'warm_history',sparse_history))
        fn=network.run_world
        if policy=='reuse':
            # Isolate re-admission of expired candidates; keep request timing, frontier and TDMA common.
            import inspect
            source=inspect.getsource(fn)
            a=source.index('        fronts=[]');b=source.index('        for p,a in zip',a)
            assert 'p.origin+a.interval>t' in source[a:b]
            local={}
            exec(source[:a]+'        fronts=[p.front for p in state]\n'+source[b:],vars(network),local)
            fn=local['run_world']
        result,_=fn(cfg,world,net,bundle,tube)
    assert abs(sum(result['ledger_s'])-180)<1e-6 and result['probes_per_user']==9000
    assert result['ten_second_G']+1e-6>=result['G_mbps']-1. # slicing an extra fraction of a 10ms slot can move its midpoint
    fields=('G_mbps','R','narrow_good_data_s','ledger_s','updates','ten_second_G','ten_second_risk','initial_center_sha256')
    write(destination,dict(scene=scene,world=w,model=name,tube=tube,warm=warm,policy=policy,margin=margin,**{k:result[k] for k in fields}))

def plan():
    jobs=[]
    def add(name,scene='new',tube='native',warm='dense',policy='strict',margin=0):
        for i in range(4):jobs.append((name,i,(scene,tube,warm,policy,margin)))
    for name in ('tuned','frozen','old'):
        for tube in ('native','lightgbm'):add(name,tube=tube)
    for name,t in [('RC','default'),('tuned','native')]:
        if name=='RC':add(name,tube=t)
        add(name,tube=t,warm='sparse')
        add(name,tube=t,policy='reuse')
        add(name,tube=t,scene='old')
    for name in ('I','P'):add(name,tube='default')
    return jobs

def bridge_plan():
    jobs=[]
    for name,scene,policy,margin in [
        ('C19','old','strict',5),('RC','old','strict',5),('C19','new','strict',0),
        ('RC','new','conservative',0),('tuned','new','conservative',0)]:
        for i in range(4):
            jobs.append((name,i,(scene,'native' if name=='tuned' else 'default','dense',policy,margin)))
    return jobs

if __name__=='__main__' and sys.argv[1]=='bridge':
    import fcntl
    index=int(sys.argv[2])
    with (OUT/'cells'/f'bridge_{index}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run_job(*bridge_plan()[index])

if __name__=='__main__' and sys.argv[1]=='cell':
    name,i,spec=plan()[int(sys.argv[2])]
    run_job(name,i,spec)

if __name__=='__main__' and sys.argv[1]=='matched':
    run_job('RC',int(sys.argv[2]),('new','lightgbm','dense','strict',0))

if __name__=='__main__' and sys.argv[1]=='gatecell':
    run_job(sys.argv[2],int(sys.argv[3]),('new','same_RC','dense','strict',0))

if __name__=='__main__' and sys.argv[1]=='oldmargin':
    run_job('C19',int(sys.argv[2]),('new','default','dense','strict',5))
