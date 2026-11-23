import numpy as np

from routeclock.config import load_config
from routeclock.radio import Radio
from routeclock.validity import ScaleHead,QUERY_H
from routeclock.evidence import Forecast
from routeclock.geometry import angles
from routeclock.horizon_gate import HorizonGate


def test_batched_worst_loss():
    r=Radio(load_config())
    rng=np.random.default_rng(42)
    dirs=rng.uniform([-np.pi,-1.5],[np.pi,1.5],(240,2))
    env=rng.uniform(0,.15,size=dirs.shape)
    for pair in r.pairs:
        np.testing.assert_allclose(r.worst_losses(dirs,env,pair),
            [r.worst_loss(a,e,pair) for a,e in zip(dirs,env)],atol=1e-9)


def test_deadline_quantile_uses_whole_prefix():
    hs=np.arange(0,10.1,.5)
    pos=np.tile([100.,0,0],(len(hs),1))
    pkg=Forecast(0,hs,{'I':pos},{},np.array([]),{})
    truth=pos.copy()
    truth[hs==1,1]=1
    truth[hs==8,2]=4
    head=ScaleHead()
    head.raw=lambda *args:np.ones((len(QUERY_H),2))
    head.calibrate([dict(package=pkg,truth_h=hs,truth=truth,features={'I':[]}) for _ in range(19)],'I')
    for h,q in head.deadline_factors.items():
        np.testing.assert_allclose(q,abs(angles(truth[hs<=h])).max(),atol=1e-14)
    assert head.factor==head.deadline_factors[10]


def test_monotone_weights_with_changing_raw_context():
    f=np.random.default_rng(13).normal(size=(21,60))
    f[:,0]=np.arange(21)*.05
    pkg=Forecast(0,f[:,0]*10,{},{},f,{'recent_spacing_s':2})
    gate=HorizonGate('g1')
    x=gate.inputs(f,pkg)
    gate.mean=np.zeros(x.shape[1]-1);gate.std=np.ones_like(gate.mean)
    gate.beta=np.linspace(-.5,.5,x.shape[1]*2)
    assert np.all(np.diff(gate.weights(x))>=0)


def test_fixed_grid_with_service_boundary():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
    from fixed_grid import fixed_scores
    from ch6_experiment import prewarm
    from routeclock.simulator import simulate
    from routeclock.world import World
    cfg=load_config()
    cfg['simulation'].update(primary_N=2,warmup_s=3.,service_s=2.,intervals_s=[1.],observation_cost_s=.0124)
    world=World(cfg,'ch6_unit',1)
    trackers=prewarm(cfg,world,.0031)
    candidate=fixed_scores(cfg,world,trackers)[-1]
    actual=simulate(cfg,world,dict(kind='fixed',seed=7,fixed=candidate),'F',service_trackers=trackers)
    np.testing.assert_allclose(candidate['net_mbps'],actual['net_mbps'],rtol=1e-12)


def test_confirmation_design(tmp_path):
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from ch6_matrix import jobs, BUDGET_POINTS, PRESSURE
    rows=jobs('confirmation',tmp_path,tmp_path/'checkpoint')
    assert len(rows)==len(set(r['out'] for r in rows))==600
    for i in range(30):
        world=[r for r in rows if r['world']==i]
        for point in BUDGET_POINTS:
            methods={r['method'] for r in world if (r['cost_ms'],r['budget_pct'])==point}
            assert methods==({'I','P','RC','F','C','CR_raw'} if i<10 else {'I','P','RC','F'})
        for point in PRESSURE:
            assert {r['method'] for r in world if (r['cost_ms'],r['budget_pct'])==point}=={'I','RC'}
        assert sum(r['scheduler'] in ('edf','uniform') for r in world)==2
    val=jobs('validation',tmp_path,tmp_path/'checkpoint')
    assert {r['world'] for r in val}==set(range(1,21))
    assert all(r['split']=='validation' for r in val)


def test_fixed_history_ignores_service_budget():
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from ch6_experiment import prewarm
    from routeclock.world import World
    cfg=load_config()
    cfg['simulation'].update(primary_N=2,warmup_s=3.,service_s=1.)
    w=World(cfg,'fixed-clock-test',2)
    def history(budget):
        cfg['simulation']['sensing_budget']=budget
        return [[o.t for o in tr.history] for tr in prewarm(cfg,w,.0031,warm_budget=.05)]
    assert history(.05)==history(.05/3)
