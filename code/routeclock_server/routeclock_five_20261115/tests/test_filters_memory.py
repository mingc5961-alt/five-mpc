import numpy as np
from routeclock.filters import PositionIMM,ProgressIMM,ou_matrices
from routeclock.memory import RouteMemory
from routeclock.world import Observation
from routeclock.config import load_config

def ob(t,p): return Observation(t,np.array(p,dtype=float),np.eye(3)*.1,0.,.2,np.zeros(4))

def test_progress_irregular_transition():
    cfg=load_config()['models']['route']; f=ProgressIMM(cfg,0,3,0)
    f.observe(3.1,3,.1,1.); result=f.forecast(np.array([0.,.5,2.,10.]))
    for w,x,P in result:
        assert np.isclose(w.sum(),1); assert np.min(np.linalg.eigvalsh(P))>=-1e-10
    assert f.t==1

def test_memory_extension_and_no_future_write():
    mem=RouteMemory(load_config()['models']['route'])
    for i in range(20): mem.update(ob(i,[300,i*2,80]),np.array([0,2,0]))
    before=[x.points.copy() for x in mem.fragments]
    mean,cov,support,std,modes=mem.forecast(np.array([0.,1.,5.]))
    assert np.isfinite(mean).all(); assert support[-1]<1
    for a,b in zip(before,mem.fragments): assert np.array_equal(a,b.points)

def test_imm_receives_radial_velocity():
    f=PositionIMM()
    for t in range(5): f.observe(ob(t,[300,t*2,80]))
    p,c=f.forecast([0,1,3]); assert p.shape==(3,3); assert np.linalg.eigvalsh(c).min()>0
