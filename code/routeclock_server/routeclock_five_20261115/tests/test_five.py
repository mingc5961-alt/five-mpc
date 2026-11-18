from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fixed_grid import fixed_scores,forecast_operator
from routeclock.config import load_config
from routeclock.filters import PositionIMM
from routeclock.world import World,Observation
from routeclock.simulator import simulate
from routeclock.foundation import ChronosAdapter

def test_exact_fixed_validation_against_event_path():
    cfg=load_config()
    cfg['simulation'].update(primary_N=2,warmup_s=3.19,service_s=2.9,intervals_s=[.5,2])
    world=World(cfg,'regression',17)
    grid=fixed_scores(cfg,world)
    for row in grid:
        result=simulate(cfg,world,dict(kind='fixed',seed=7,fixed=row),'F')
        np.testing.assert_allclose(result['net_mbps'],row['net_mbps'],rtol=1e-10,atol=1e-8)

def test_unobserved_first_moments():
    cfg=load_config(local=True);w=World(cfg,'regression',2,n=1);f=PositionIMM()
    for t in [1.,2.,2.43,3.72]:f.observe(w.observe(0,t))
    predicted=(forecast_operator()@(f.weights[:,None]*f.x).ravel()).reshape(21,3)
    np.testing.assert_allclose(predicted,f.forecast(np.arange(21)*.5)[0],atol=1e-9)

def test_equal_observable_channels():
    history=[Observation(t,np.array([300.,t,80]),np.eye(3)*2.,t+1,.2,np.zeros(4)) for t in [0,1,2]]
    p=np.tile(history[-1].position,(21,1));s=np.ones_like(p);support=np.ones(21)
    a,_=ChronosAdapter.inputs(history,[],p,s,support,False);b,_=ChronosAdapter.inputs(history,[],p,s,support,True)
    for key,v in a['past_covariates'].items():np.testing.assert_equal(b['past_covariates'][key],v)
    assert a['past_covariates']['radial_velocity'][-1]==3
    assert a['past_covariates']['position_cov_22'][-1]==2
    assert 'future_covariates' not in a
    assert all(k.startswith('route_') for k in b['future_covariates'])

def test_paired_five_model_worklist():
    from five_campaign import jobs
    expected={'main':180,'actions':90,'reuse':120}
    for block,count in expected.items():
        batch=jobs(block)
        assert len(batch)==len(set(batch))==count
    for i in range(30):
        assert {m for j,m,_,_ in jobs('main') if j==i}=={'F','I','P','RC','C','CR_raw'}
        assert {a for j,_,a,_ in jobs('actions') if j==i}=={'width_only','interval_only','fixed'}

