import sys,pickle,fcntl
from protocol_local import OLD,OUT,log
import replay_new as prev
from features import common,rc_features,apply_gate
from fit_heads import predict
import numpy as np
original=prev.Model
class Model(original):
    def __init__(self,arm,world,net,head):
        super().__init__('joint-refit' if arm.startswith('v2') else arm,world,net,head)
        if arm.startswith('v2'):
            from mileage import selected
            self.learn=selected('cpu')
        self.variant=arm
    def batch(self,hs):
        if not self.variant.startswith('v2'):return super().batch(hs)
        from mileage import infer
        ps=self.history_packages(hs);rows=[];zs=[]
        for h,p in zip(hs,ps,strict=True):
            apply_gate(p,self.gate,p.features,'RC','P')
            tr=self.track[id(h)];d=prev.pack(h,tr.memory,tr.imm,p)
            rows.append(d);zs.append(rc_features(p,p.features))
        cs=infer(self.learn,rows,hs);sc=predict(self.head,np.stack([common(h,c) for h,c in zip(hs,cs)]),np.stack(zs))
        return list(zip(cs,sc))
head_old=prev.head_for
def head(arm):
    if arm=='v2-refit':
        hd=pickle.loads((OUT/'heads/v2/special.pkl').read_bytes());return hd,hd['factors']
    if arm in ('joint-refit','v2-transfer'):
        hd=pickle.loads((OLD/'heads/joint/special.pkl').read_bytes());return hd,hd['factors']
    return head_old(arm)
def setup():
    prev.rr.setup();prev.ROOT=OUT;prev.runtime.OUT=OUT/'cache'
    prev.Model=Model;prev.head_for=head;prev.log=lambda s:log(s.replace('/210','/150'))
if __name__=='__main__':
    setup();eng={}
    for w in range(50,60):
        for b in 'TML':
            for arm in sys.argv[1:]:prev.run(arm,w,b,eng)
