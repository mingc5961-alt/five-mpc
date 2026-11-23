import hashlib
import json
import pickle
import sys
import time
import numpy as np
from protocol import OUT,ROOT,PRIOR,NS,SEED,BUDGETS,KINDS,write,log
from features import Advanced,common,rc_features,rc_params,apply_gate
from fit_heads import predict
from parallel_frontier import frontier
from runtime import remaining_runs as runtime
from refine import network,world_cache
from routeclock.config import load_config
from routeclock.radio import Radio

def reference(model,w,b):
    if model=='RC':
        return PRIOR/'baseline_cells'/f'{w:03}_{b}_RC.json'
    method='native' if model=='Chronos-2' else 'fixed'
    cell=f'{w:03}_{b}_{method}'
    replay=PRIOR/'先进模型硬件复核'/model/(cell+'.json')
    return replay if replay.exists() else ROOT/'results/development'/model/'7'/cell/'complete.json'

class ForecastTube:
    def __init__(self,model,engine,head,world):
        self.engine=engine
        self.model=model
        self.head=head
        self.private=head['kind']=='special'
        self.trackers={}
        if model=='RC':
            self.warm=runtime.warm_trackers(world)
            self.gate=rc_params()['gate']
    def batch(self,histories):
        if self.model=='RC':
            rows=[]
            if not self.trackers:
                for h,tr in zip(histories,self.warm,strict=True):
                    assert h[-1].t==tr.history[-1].t and len(h)==len(tr.history)
                    self.trackers[id(h)]=tr
            for h in histories:
                tr=self.trackers[id(h)]
                for obs in h[len(tr.history):]:
                    tr.observe(obs);tr.issue()
                p=tr.snapshots[-1];apply_gate(p,self.gate,p.features,'RC','P')
                rows.append((p.centers['RC'],rc_features(p,p.features) if self.private else None))
        else:
            rows=self.engine.extract(histories,self.private)
        x=np.stack([common(h,c) for h,(c,_) in zip(histories,rows,strict=True)])
        z=np.stack([v for _,v in rows]) if self.private else None
        widths=predict(self.head,x,z)
        return [(c,width) for (c,_),width in zip(rows,widths,strict=True)]

def run(model,kind,start=0,stop=10):
    network.frontier=frontier
    world_cache.BASE=ROOT/'work/baseline_cache';runtime.OUT=PRIOR/'cache'
    file=OUT/'heads'/model/(kind+'.pkl')
    blob=file.read_bytes();digest=hashlib.sha256(blob).hexdigest();head=pickle.loads(blob)
    engine=Advanced(model) if model!='RC' else None
    directory=OUT/'cells'/model/kind
    for w in range(start,stop):
        cfg=load_config();cfg['simulation']['sensing_budget']=runtime.BUDGET['T']
        world=world_cache.paired_world(cfg,NS,w)
        for b in BUDGETS:
            dest=directory/f'{w:03}_{b}.json'
            if dest.exists():
                assert json.loads(dest.read_text())['head_sha256']==digest
                continue
            cfg=load_config();rho=runtime.BUDGET[b];cfg['simulation']['sensing_budget']=rho
            radio=Radio(cfg);radio.threshold-=5
            net=ForecastTube(model,engine,head,world)
            began=time.monotonic()
            with runtime.controller(radio,10.):
                r,_=network.run_world(cfg,world,net,{'factors':{kind:head['factors']}},kind)
            ref=json.loads(reference(model,w,b).read_text())
            assert ref['namespace']==NS and r['initial_center_sha256']==ref['result']['initial_center_sha256']
            assert abs(sum(r['ledger_s'])-180)<1e-6 and abs(r['ledger_s'][0]-r['updates']*.0031)<1e-6
            assert r['ledger_s'][0]<=rho*180+.0031001 and abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-7
            assert 0<=r['unavailable_fraction']<=r['R']<=1 and len(r['bits'])==16 and r['probes_per_user']==9000
            write(dest,dict(model=model,head=kind,world=w,budget=b,namespace=NS,seed=SEED,
                            head_sha256=digest,mcs_margin_db=0,risk_scale=1,deadline_s=10,
                            result=r,elapsed_s=time.monotonic()-began))
            print(w,b,round(r['G_mbps'],2),round(r['R']*100,3),flush=True)
        if w in (2,5,9):
            finished=len(list((OUT/'cells').glob('*/*/*.json')))
            log(f'{model} {kind} 已通过世界 {w}，当前八配置合计 {finished}/240 场闭环，各回执已核对预热中心和时间账本。')
if __name__=='__main__':
    run(sys.argv[1],sys.argv[2],*map(int,sys.argv[3:]))
