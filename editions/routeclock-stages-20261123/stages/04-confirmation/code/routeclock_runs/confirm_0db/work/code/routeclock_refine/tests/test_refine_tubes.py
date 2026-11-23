import numpy as np
from routeclock.config import load_config
from routeclock.radio import Radio
from routeclock.scheduler import Action
from types import SimpleNamespace
from refine.network import Ledger
from refine.tubes import scores

def test_cut_data_slot_at_expiry_and_count_idle_users():
    cfg=load_config()
    cfg['simulation'].update(tdma_quantum_s=.01,beam_switch_s=.0001)
    times=np.arange(8)*.02
    pos=np.tile([300.,0.,50.],(2,len(times),1))
    world=SimpleNamespace(n=2,dt=.02,positions=pos,shadows=np.zeros((2,len(times))))
    ledger=Ledger(cfg,world,Radio(cfg),0.,.04)
    state=[SimpleNamespace(action=Action((8,8),.006,1.,0.),origin=0.,center=pos[0,:1].repeat(21,axis=0)),
           SimpleNamespace(action=None,origin=0.)]
    ledger.advance(0.,.04,state,False)
    result=ledger.result(.0031,.05/3)
    np.testing.assert_allclose(result['ledger_s'],[0.,.0004,.0059,.0337])
    assert result['R']==1. and result['probes_per_user']==2
    assert result['bits'][0]>0 and result['bits'][1]==0

def test_dense_joint_prefix_detects_error_between_knots():
    err=np.zeros((1,501,2));err[:,33,1]=2.
    raw=np.ones((1,21,2));measured=scores(err,raw)
    np.testing.assert_equal(measured[0],[0,2,2,2,2,2,2])
