import sys
sys.path.insert(0, '/data/chen/formal_execution')
from resource_control import admission_order


def job(name, head='lightgbm', deps=(), kind='network'):
    return dict(id=name,kind=kind,meta=dict(head=head),dependencies=list(deps))


def test_balance_ready_families_preserves_per_family_order():
    jobs=[job('c1'),job('c2'),job('t1','tabpfn'),job('t2','tabpfn')]
    states={j['id']:'pending' for j in jobs}
    assert [j['id'] for j in admission_order(jobs,states,[],{'balance_network_families':True})]==['c1','t1','c2','t2']
    assert [j['id'] for j in jobs]==['c1','c2','t1','t2']


def test_inflight_families_and_prerequisite_priority():
    jobs=[job('c1'),job('c2'),job('t1','tabpfn'),job('t2','tabpfn'),job('summary',kind='summary')]
    states={j['id']:'pending' for j in jobs}
    running=[job('r1'),job('r2')]
    order=admission_order(jobs,states,running,{'balance_network_families':True})
    assert [j['id'] for j in order[:3]]==['summary','t1','t2']


def test_unready_or_completed_jobs_not_scheduled():
    jobs=[job('c'),job('t','tabpfn',deps=['eval']),job('eval',kind='train'),job('done')]
    states=dict(c='pending',t='pending',eval='pending',done='passed')
    assert [j['id'] for j in admission_order(jobs,states,[],{'balance_network_families':True})]==['eval','c']


def test_disabled_policy_retains_original_order():
    jobs=[job('a'),job('b')]
    assert admission_order(jobs,{},[],None) is jobs
    assert admission_order(jobs,{},[],{}) is jobs
