"""覆盖预算、滚动时域、10 秒边界和多机通信时间竞争的数值回归测试。"""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linprog
from mpc.control import slow_plan,fast_plan,sensing_clock,one_step_allocation
from mpc.data import GRID
from mpc.risk import Profile

def profile(origin, high, decay):
    rates=np.column_stack([high-decay*np.minimum(GRID,4),np.full(len(GRID),high*.65)])
    return Profile(SimpleNamespace(origin=origin),np.minimum(GRID/10,1),rates)

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(dt=.062,target_network_mbps=15.,deficit_weight=.8,
                    min_airtime_fraction=.25,search_width=12)
        self.profiles=[profile(0,20,3),profile(-2,21,4)]

    def test_budget_each_prefix(self):
        for budget in [.00959,.02,.05,.08]:
            active=sensing_clock(192,.062,budget,.0031)
            self.assertTrue(np.all(np.cumsum(active)*.0031<=np.arange(1,193)*.062*budget+1e-9))
        self.assertEqual(sensing_clock(192,.062,.05,.0031).sum(),192)

    def test_zero_rate_after_horizon(self):
        self.assertGreater(self.profiles[0].at(10.)[0],0)
        self.assertEqual(self.profiles[0].at(10.01)[0],0)

    def test_slow_respects_unavailable_slots_and_can_wait(self):
        a,_=slow_plan(self.profiles,1,np.zeros(2),3,self.c,.0031,.0001,[False,True,False])
        self.assertEqual((a[0],a[2]),(-1,-1))
        flat=[profile(0,20,0),profile(0,20,0)]
        a,_=slow_plan(flat,1,np.zeros(2),3,self.c,.0031,.0001)
        self.assertEqual(a,[-1]*3)

    def test_one_tick_solver_matches_lp(self):
        rng=np.random.default_rng(7)
        for _ in range(12):
            r=rng.uniform(0,35,size=4)
            t=.05
            due=rng.uniform(0,1,size=4)
            w=.8
            bits,debt=one_step_allocation(r[None,:],np.array([t]),due[None,:],.25,w)
            sol=linprog(np.r_[-r,np.full(4,w)],A_ub=np.c_[-np.diag(r),-np.eye(4)],
                        b_ub=-due,A_eq=[np.r_[np.ones(4),np.zeros(4)]],b_eq=[t],
                        bounds=[(.25*t/4,None)]*4+[(0,None)]*4)
            self.assertAlmostEqual(bits[0]-w*debt.sum(),-sol.fun,places=8)

    def test_fast_first_action_sums_to_time(self):
        a,b=fast_plan(self.profiles,1,np.array([.5,.1]),[-1,0,1],self.c,.0031,.0001)
        self.assertAlmostEqual(a.sum(),.062-.0002,places=9)
        self.assertTrue(np.all(a>=.25*(.062-.0002)/2-1e-9))
        self.assertEqual(len(b),2)

    def test_fast_lookahead_changes_time_allocation(self):
        # 甲下一轮过期，乙两轮均可服务。终端欠账使快层在甲仍可传时提前给它时间。
        a=profile(-9.94,20,0)
        b=profile(0,22,0)
        c=dict(self.c,target_network_mbps=0.,deficit_weight=2.)
        q=np.array([.65,2.])
        short,_=fast_plan([a,b],0,q,[-1],c,.0031,.0001)
        long,_=fast_plan([a,b],0,q,[-1,-1],c,.0031,.0001)
        self.assertGreater(long[0],short[0]+.001)

if __name__=="__main__":
    unittest.main()
