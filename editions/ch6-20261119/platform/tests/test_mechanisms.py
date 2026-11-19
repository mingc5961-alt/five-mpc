import numpy as np
from routeclock.config import load_config
from routeclock.memory import RouteMemory,Fragment
from routeclock.filters import ProgressIMM
from routeclock.radio import Radio,array_gain
from routeclock.scheduler import frontier,choose_actions
from routeclock.evidence import Forecast,Evidence
from routeclock.world import Observation
from routeclock.predictors import Tracker,Gate

def test_action_restriction_before_pruning():
    cfg=load_config();r=Radio(cfg);hs=np.arange(0,10.001,.5);p=np.tile([300.,0.,80.],(len(hs),1))
    pkg=Forecast(0,hs,{'I':p},{},np.zeros((len(hs),56)),{})
    fs=frontier(pkg,'I',np.zeros((len(hs),2)),r,cfg)
    assert choose_actions([fs],.0031,.05,width_only=True)[0] is not None
    assert choose_actions([fs],.0031,.05,interval_only=True)[0].pair==(16,16)

def test_angular_bound_against_dense_joint_grid():
    r=Radio(load_config());rng=np.random.default_rng(19)
    for _ in range(20):
        a=rng.uniform([-.6,.1],[.6,.6]);e=rng.uniform(.001,.04,2);pair=(16,32)
        aa,ee=np.meshgrid(np.linspace(-1,1,101),np.linspace(-1,1,101));truth=a+np.c_[aa.ravel(),ee.ravel()]*e
        actual=-10*np.log10(array_gain(a,truth,pair).min()/np.prod(pair))
        assert r.worst_loss(a,e,pair)+1e-8>=actual

def test_same_fragment_branch_hypotheses_propagate():
    cfg=load_config()['models']['route'];m=RouteMemory(cfg)
    m.fragments=[Fragment(np.array([[0.,0,0],[10.,0,0],[10.,10,0],[0.,10,0]]))];m.active=0;m.s=5
    m.progress=ProgressIMM(cfg,5,1,0);m.last=Observation(0,np.array([5.,0,0]),np.eye(3)*.01,0,.2,np.zeros(4))
    m.matches=[(0,5,.5),(0,25,.5)]
    mean,cov,*_=m.forecast([0])
    assert mean[0,1]>4;assert cov[0,1,1]>20

def test_realized_evidence_does_not_use_future_or_refit():
    e=Evidence();hs=np.array([0.,1.,2.]);center=np.array([[300.,0,80],[300.,1,80],[300.,2,80]])
    pkg=Forecast(0,hs,{'P':center.copy(),'I':center.copy()},{},np.zeros((3,56)),{})
    e.issue(pkg);assert e.summary(.5)[2]==0
    e.observe(Observation(1,np.array([300.,3,80]),np.eye(3),0,.2,np.zeros(4)))
    assert e.summary(1)[2]>0;assert np.array_equal(pkg.centers['P'],center)

def test_gate_angular_objective_learns_useful_expert():
    rng=np.random.default_rng(7);X=rng.normal(size=(80,16));y=np.c_[np.full(80,300.),rng.normal(size=80)*10,np.full(80,80.)]
    good=y.copy();bad=y+np.array([0,30,0]);g=Gate().fit(X,good,bad,y,l2=.0001)
    assert g.weights(X).mean()>.9

def test_harmonic_and_snapshot_causality():
    t=Tracker(load_config())
    for k in range(5):t.observe(Observation(k,np.array([300.,k,80]),np.eye(3),0,.2,np.zeros(4)))
    pkg=t.issue();old={k:v.copy() for k,v in pkg.centers.items()}
    t.observe(Observation(5,np.array([300.,20,80]),np.eye(3),0,.2,np.zeros(4)));t.issue()
    for k,v in old.items():assert np.array_equal(v,pkg.centers[k])
    assert t.evidence.summary(5).shape==(28,)
    assert t.evidence.summary(5)[25]>0
    assert pkg.features.shape[1]==60

def test_midroute_departure_is_not_terminal_transition():
    cfg=load_config()['models']['route'];m=RouteMemory(cfg)
    m.fragments=[Fragment(np.array([[300.,0,80],[400.,0,80]]))];m.active=0;m.s=40
    m.progress=ProgressIMM(cfg,40,2,0);m.matches=[(0,40,1.)]
    m.last=Observation(0,np.array([340.,0,80]),np.eye(3)*.1,0,.2,np.zeros(4))
    m.update(Observation(1,np.array([340.,10,80]),np.eye(3)*.1,0,.2,np.zeros(4)),np.array([0.,2.,0.]))
    assert (0,1) not in m.connections
    assert np.allclose(m.map_distribution(0,101)[0][0],[401.,0,80])

def test_connection_bridge_consumes_distance():
    m=RouteMemory(load_config()['models']['route']);m.fragments=[Fragment(np.array([[0.,0,0],[10.,0,0]])),Fragment(np.array([[10.,10,0],[20.,10,0]]))]
    m.connections={(0,1):(1,0.)}
    assert np.allclose(m.map_distribution(0,15)[0][0],[10.,5,0])

def test_point_progress_ablation_collapses_mode_variance():
    cfg=load_config()['models']['route'];m=RouteMemory(cfg);m.fragments=[Fragment(np.array([[300.,0,80],[400.,0,80]]))];m.active=0;m.s=30
    m.progress=ProgressIMM(cfg,30,4,0);m.last=Observation(0,np.array([330.,0,80]),np.eye(3)*.1,0,.2,np.zeros(4));m.matches=[(0,30,1.)]
    _,cov,_,std,_=m.forecast([0,5],point=True)
    assert np.allclose(std,0);assert np.allclose(cov[-1],m.last.covariance)
