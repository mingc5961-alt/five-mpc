"""Frozen predictors on the same paid observations and paired physical inputs."""
import json,pickle,sys
from functools import lru_cache
import numpy as np
from history import ROOT
from paired_grid import BASE
from paired_models import Predictor as Calibrated

AUX=BASE/'auxiliary/models'
@lru_cache
def auxiliary():
    with (AUX/'raft_width.pkl').open('rb') as f:tree=pickle.load(f)
    factors=json.loads((AUX/'calibration.json').read_bytes())['factors']
    return tree,{k:np.asarray(v) for k,v in factors.items()}
@lru_cache
def raft_model():
    import raft_rebuild
    current=raft_rebuild.RUN
    try:
        raft_rebuild.RUN=BASE/'auxiliary'
        return raft_rebuild.Raft()
    finally:raft_rebuild.RUN=current
class Model:
    def __init__(self,w,kind):
        import grurun
        import paid_full as pf
        self.kind=kind;self.h=[[] for _ in range(w.n)]
        if kind in ('Acal','M','Fstar'):
            self.base=Calibrated(w,kind);self.factors=self.base.factors
        elif kind in ('Aorig','F0orig','Chronos'):
            self.base=grurun.Predictor(w.cfg,'C' if kind=='Chronos' else 'G1')
            if kind=='F0orig':self.base.arm='Static'
            self.factors=self.base.factors
        else:
            from routeclock.filters import PositionIMM
            self.width,self.factors_by_kind=auxiliary();self.factors=self.factors_by_kind[kind]
            self.base=[PositionIMM() for _ in range(w.n)] if kind=='IMM' else raft_model()
    def predict(self,u,o):
        import paid_full as pf
        from routeclock.geometry import angular_jacobian
        if self.kind in ('Acal','M','Fstar'):return self.base.predict(u,o)
        h=self.h[u];h.append(o)
        if self.kind in ('Aorig','F0orig','Chronos'):return self.base.update(u,h)
        if self.kind=='IMM':
            tracker=self.base[u];tracker.observe(o);c,cov=tracker.forecast(pf.GRID)
            j=np.array([angular_jacobian(p) for p in c])
            width=np.sqrt(np.maximum(1e-12,np.einsum('hij,hjk,hik->hi',j,cov,j)))
            return c,width
        c=self.base.predict([h])[0]
        diff=np.abs(pf.difference(pf.angle(c),pf.angle(o.position)))
        f=np.column_stack([np.repeat(pf.GRID,2),np.tile([0.,1.],21),diff.ravel(),np.full(42,np.log1p(len(h))),
            np.full(42,o.t-h[-2].t if len(h)>1 else 0.),np.full(42,np.linalg.norm(o.position)),np.full(42,np.trace(o.covariance))])
        return c,np.exp(np.clip(self.width.predict(f,num_threads=1),-25,3)).reshape(21,2)
