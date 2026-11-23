import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ch6_dispatch import Admission
from ch6_cache import prepare
from ch6_experiment import prewarm
from routeclock.config import load_config
from routeclock.world import World
from routeclock.simulator import simulate


def test_prioritizes_gpu_and_respects_reserved_jobs(tmp_path):
    bundle=tmp_path/'bundle'
    bundle.touch()
    def row(method,key):
        return dict(method=method,out=key,argv=['--bundle',str(bundle)])
    rows=deque([row('RC','cpu'),row('C','busy'),row('CR_raw','available')])
    p=Admission()
    assert p.next_job(rows,0,{'busy'})['out']=='available'
    assert p.next_job(rows,p.gpu_slots,{'busy'})['out']=='cpu'
    assert p.next_job(rows,0,{'busy'}) is None


def test_cache_exact_service_results_across_service_costs(tmp_path):
    cfg=load_config()
    cfg['simulation'].update(primary_N=3,warmup_s=5.,service_s=4.)
    args=('equivalence',7,.0031,.05,None,False,prewarm,tmp_path/'cache')
    _,_,miss=prepare(cfg,*args)
    assert not miss['warm']['hit']
    cfg['simulation'].update(observation_cost_s=.0093,sensing_budget=.025)
    cached,states,hit=prepare(cfg,*args)
    assert hit['world']['hit'] and hit['warm']['hit']
    baseline=World(cfg,'equivalence',7)
    fresh=prewarm(cfg,baseline,.0031,warm_budget=.05)
    bundle=dict(kind='fixed',seed=7,fixed={'pair':[32,32],'interval':2.})
    a=simulate(cfg,baseline,bundle,'F',out=tmp_path/'fresh',service_trackers=fresh)
    b=simulate(cfg,cached,bundle,'F',out=tmp_path/'cached',service_trackers=states)
    for key in ['net_mbps','hold_failures','sensing_s','narrow_share','observations_total','worst10_mbps']:
        np.testing.assert_equal(a[key],b[key])
    assert (tmp_path/'cached'/'events.json').read_bytes()==(tmp_path/'fresh'/'events.json').read_bytes()
