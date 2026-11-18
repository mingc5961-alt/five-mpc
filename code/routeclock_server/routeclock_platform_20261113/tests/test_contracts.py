import numpy as np
from routeclock.scheduler import TokenBucket,Action,choose_actions
from routeclock.validity import finite_quantile,feature_rows,QUERY_H
from routeclock.experiments import experiment_matrix
from routeclock.evidence import Evidence,Forecast
from routeclock.world import Observation
from routeclock.foundation import ChronosAdapter

def test_budget_and_resource_selection():
    b=TokenBucket(.0031,.02);t=0
    for k in range(10):t=b.ready(t);b.consume(t)
    assert t>=9*.0031/.02-1e-10
    fs=[[Action((32,32),1,100,0),Action((16,16),5,90,0)]]*3
    actions=choose_actions(fs,.0031,.002)
    assert sum(.0031/a.interval for a in actions)<=.002

def test_calibration_rank_and_axis_encoding():
    assert finite_quantile(np.arange(10),.1)==9
    rows=feature_rows(np.ones((21,60)),np.arange(0,10.001,.5))
    assert rows.shape==(16,63);assert np.array_equal(rows[0,-2:],[1,0]);assert np.array_equal(rows[1,-2:],[0,1])

def test_matrix():
    jobs=experiment_matrix();assert len(jobs)==830;assert max(x['N'] for x in jobs)==32
    assert len([x for x in jobs if x['block']=='B_Z'])==20
    from routeclock.experiments import single_action_matrix
    assert len(single_action_matrix())==80

def test_no_hindsight_snapshots_and_route_omission():
    history=[Observation(t,np.array([300.,t,80]),np.eye(3),0,.2,np.zeros(4)) for t in range(4)]
    hs=np.arange(0,10.001,.5);p=np.tile(history[-1].position,(21,1));cov=np.tile(np.eye(3),(21,1,1))
    pkg=Forecast(3,hs,{'P':p},{'P':cov},np.empty((21,0)),{'support':np.ones(21)})
    c,_=ChronosAdapter.inputs(history,[pkg],p,np.ones_like(p),np.ones(21),False)
    assert 'future_covariates' not in c;assert not any('route' in k for k in c['past_covariates'])
    cr,_=ChronosAdapter.inputs(history,[pkg],p,np.ones_like(p),np.ones(21),True)
    assert np.isnan(cr['past_covariates']['route_x'][:-1]).all()

def test_calibration_has_same_causal_observation_protocol():
    from routeclock.dataset import observation_schedule
    anchors=[7.8,9.84,12.]
    test=observation_schedule(anchors,anchors,5,np.random.default_rng(7))
    for chosen in anchors:
        cal=observation_schedule(anchors,[chosen],5,np.random.default_rng(7))
        assert cal==[t for t in test if t<=chosen]
