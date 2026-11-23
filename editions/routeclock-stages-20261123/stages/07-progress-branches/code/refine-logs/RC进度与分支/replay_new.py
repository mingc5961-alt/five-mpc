import sys,json,pickle,time,fcntl,hashlib
from unittest.mock import patch
from core import ROOT,log,pack
import numpy as np
import replay_route as rr
from features import Advanced,common,rc_features,rc_params,apply_gate
from fit_heads import predict
from routeclock.radio import Radio
from routeclock.config import load_config
from refine import network
from runtime import remaining_runs as runtime
WORLDS=range(40,50)
BUDGETS=('T','M','L')
ARMS=('RC','Chronos-2','VECTOR','RAFT','Chronos-native','joint-transfer','joint-refit')
def setup():
    rr.setup()
    runtime.OUT=ROOT/'cache';runtime.OUT.mkdir(exist_ok=True)
def head_for(arm):
    if arm=='Chronos-native':return None,rr.CN['factors']['native']
    if arm=='joint-refit':p=ROOT/'heads/joint/special.pkl'
    else:p=rr.SPECIAL/'heads'/('RC' if arm.startswith('joint') else arm)/'special.pkl'
    head=pickle.loads(p.read_bytes());return head,head['factors']
class Model(rr.Model):
    def __init__(self,arm,world,net,head):
        self.arm=arm;self.net=net;self.head=head;self.track={};self.only_route=False
        if arm=='RC' or arm.startswith('joint'):
            self.warm=runtime.warm_trackers(world);self.gate=rc_params()['gate']
        if arm.startswith('joint'):
            from learning import load
            self.learn=load('cpu')
    def batch(self,histories):
        if self.arm=='Chronos-native':return self.net.net.batch(histories)
        if self.arm=='RC' or self.arm.startswith('joint'):
            ps=self.history_packages(histories);rows=[];inputs=[]
            for h,p in zip(histories,ps,strict=True):
                apply_gate(p,self.gate,p.features,'RC','P')
                tr=self.track[id(h)]
                if self.arm.startswith('joint'):inputs.append(pack(h,tr.memory,tr.imm,p))
                rows.append((p.centers['RC'],rc_features(p,p.features)))
            if inputs:
                from learning import infer
                cs=infer(self.learn,inputs,'joint');rows=[(c,r[1]) for c,r in zip(cs,rows,strict=True)]
        else:rows=self.net.extract(histories,True)
        scales=predict(self.head,np.stack([common(h,c) for h,(c,z) in zip(histories,rows,strict=True)]),np.stack([z for c,z in rows]))
        return [(r[0],s) for r,s in zip(rows,scales,strict=True)]
def run(arm,w,b,engines):
    dest=ROOT/'cells'/f'{w}_{b}_{arm}.json'
    if dest.exists():return
    with dest.with_suffix('.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if dest.exists():return
        start=time.monotonic();world=rr.get_world(w)
        if arm in ('Chronos-native','Chronos-2','VECTOR','RAFT'):
            key='Chronos-2' if arm=='Chronos-native' else arm
            if key not in engines:
                engines[key]=Advanced(key)
                if key=='Chronos-2':
                    from graph_encoder import enable
                    enable(engines[key].net.net)
            engine=engines[key]
        else:engine=None
        head,factors=head_for(arm);cfg=load_config();rho=runtime.BUDGET[b];cfg['simulation']['sensing_budget']=rho
        radio=Radio(cfg);radio.threshold-=5
        with rr.memoized_arc():
            model=Model(arm,world,engine,head)
            with runtime.controller(radio,10.),patch.object(network,'Ledger',rr.HoldLedger):
                r,_=network.run_world(cfg,world,model,{'factors':{arm:factors}},arm)
        assert abs(sum(r['ledger_s'])-180)<1e-6 and abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-7
        assert abs(r['ledger_s'][0]-.0031*r['updates'])<1e-6 and r['ledger_s'][0]<=rho*180+.0031001
        assert r['probes_per_user']==9000 and 0<=r['unavailable_fraction']<=r['R']<=1
        rr.write(dest,dict(arm=arm,world=w,budget=b,namespace=rr.NS,result=r,elapsed_s=time.monotonic()-start))
        log(f'闭环 {len(list(dest.parent.glob("*.json")))}/210，世界 {w} {b} {arm}，吞吐 {r["G_mbps"]:.2f} Mbps、风险 {r["R"]:.2%}、平均保持 {r["hold_mean_s"]:.3f}s、感知 {r["ledger_s"][0]:.3f}s。')
def main(arms):
    setup();(ROOT/'cells').mkdir(exist_ok=True);engines={}
    for w in WORLDS:
        for b in BUDGETS:
            for arm in arms:run(arm,w,b,engines)
if __name__=='__main__':main(sys.argv[1:])
