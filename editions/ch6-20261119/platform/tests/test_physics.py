import numpy as np
from routeclock.config import load_config
from routeclock.world import World,GROUPS
from routeclock.radio import Radio,array_gain

def test_world_pairing_and_limits():
    c=load_config(True); c['simulation']['warmup_s']=0;c['simulation']['service_s']=3
    a=World(c,'test/physics',1,n=6,groups=GROUPS); b=World(c,'test/physics',1,n=6,groups=GROUPS)
    assert np.array_equal(a.positions,b.positions)
    assert np.linalg.norm(a.velocities,axis=-1).max()<=12+1e-8
    assert np.linalg.norm(a.accelerations,axis=-1).max()<=4+1e-8
    assert np.linalg.norm(np.diff(a.accelerations,axis=1)/a.dt,axis=-1).max()<=8+1e-6
    assert np.array_equal(a.observe(0,1.123).raw,a.observe(0,1.123).raw)

def test_radio_peak_and_gain():
    assert np.isclose(array_gain([.1,.2],[.1,.2],(32,16)),512)
    r=Radio(load_config())
    assert r.worst_loss(np.array([.1,.2]),np.array([0.,0.]),(32,32))<1e-8
    assert r.worst_loss(np.array([.1,.2]),np.array([.05,.05]),(32,32))>3
