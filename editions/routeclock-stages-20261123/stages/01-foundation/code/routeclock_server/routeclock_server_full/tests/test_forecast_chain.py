"""从头训练链的关键回归：单机门控拟合、方向图几何界、快慢层都遵守误差管。"""
import unittest
from types import SimpleNamespace
import numpy as np
from routeclock.predictors import Gate
from routeclock.geometry import angles
from routeclock.radio import Radio
from mpc.data import GRID,load_config
from mpc.forecast_training import beam_deadlines
from mpc.risk import Profile
from mpc.control import rate_table

class ForecastTests(unittest.TestCase):
    def test_gate_fits_angular_target(self):
        rng=np.random.default_rng(34)
        x=rng.normal(size=(200,16))
        p=np.tile([110.,13.,40.],(200,1))
        i=np.tile([110.,-13.,40.],(200,1))
        w=(1/(1+np.exp(-3*x[:,0])))[:,None]
        y=w*p+(1-w)*i
        g=Gate().fit(x,p,i,y)
        self.assertTrue(g.fit_status["success"])
        self.assertLess(np.mean((angles(g.predict(x,p,i))-angles(y))**2),
                        np.mean((angles((p+i)/2)-angles(y))**2)*.25)

    def test_deadlines_match_physical_bound(self):
        radio=Radio(load_config())
        rc=np.column_stack([-90+GRID,np.linspace(-15,15,len(GRID)),50+GRID])
        pkg=SimpleNamespace(centers={"RC":rc},hs=GRID)
        env=np.column_stack([GRID*.019+.002,GRID*.012+.001])
        expected=[]
        for pair in radio.pairs:
            safe=[radio.worst_loss(a,e,pair)<=3 for a,e in zip(angles(rc),env)]
            expected.append(np.where(np.logical_and.accumulate(safe),GRID,-1).max())
        np.testing.assert_equal(beam_deadlines(pkg,env,radio.pairs,3),expected)

    def test_bound_enters_both_controllers(self):
        # 窄束速率较高但仅允许 0.25 s，宽束允许 1 s；到期后两层必须同步改选。
        pr=Profile(SimpleNamespace(origin=0.),np.zeros(len(GRID)),
                   np.tile([4.,12.],(len(GRID),1)),np.array([1.,.25]))
        self.assertEqual(pr.at(.3)[:2],(4.,0))
        self.assertEqual(pr.at(1.1)[0],0.)
        tbl=rate_table([pr],.29,1,.02,.001)
        self.assertEqual(tbl[0,0,0],4.)
        self.assertEqual(tbl[0,0,1],12.)
