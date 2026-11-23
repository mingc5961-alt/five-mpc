import copy
import json

import pytest

import resource_control as r


@pytest.fixture
def policy():
    return dict(cpu_capacity=12, cpu_network_cost=1.25, gpu_network_cpu_cost=2,
                tabpfn_train_cpu_cost=2, max_cpu_network_jobs=8, memory_headroom_gib=6,
                cpu_network_memory_gib=1.5, gpu_network_memory_gib=3,
                tabpfn_memory_gib=6, other_memory_gib=4)


def job(id='cpu', gpu=0, head='lightgbm'):
    return dict(id=id, kind='network', cpu=4, gpu=gpu, meta=dict(head=head))


def test_cpu_network_parallelism_respects_both_limits(policy):
    j=job()
    assert r.can_admit(j,policy,10.75,12,2,2,7,20*r.GIB,26*r.GIB)
    assert not r.can_admit(j,policy,12,12,2,2,8,20*r.GIB,26*r.GIB)
    assert not r.can_admit(j,policy,4,12,0,2,8,10*r.GIB,26*r.GIB)


def test_gpu_exclusivity_is_preserved(policy):
    assert not r.can_admit(job(gpu=1),policy,2,12,2,2,0,6*r.GIB,26*r.GIB)
    assert not r.can_admit(job(gpu=2,head='tabpfn'),policy,2,12,1,2,0,6*r.GIB,26*r.GIB)
    assert r.can_admit(job(),policy,2,12,2,2,0,6*r.GIB,26*r.GIB)


def test_startup_growth_is_reserved_without_resource_file(tmp_path,policy):
    active=[job('a'),job('b')]
    folder=tmp_path/'jobs/a';folder.mkdir(parents=True)
    (folder/'resource.json').write_text(json.dumps(dict(rss_bytes=.5*r.GIB)))
    ceiling,committed=r.memory_budget(tmp_path,active,policy,
        dict(memory_limit_bytes=32*r.GIB,memory_current_bytes=22*r.GIB))
    assert ceiling==26*r.GIB
    assert committed==24.5*r.GIB
    assert r.can_admit(job('c'),policy,2.5,12,0,2,2,committed,ceiling)
    committed+=r.memory_reservation(job('c'),policy)
    assert not r.can_admit(job('d'),policy,3.75,12,0,2,3,committed,ceiling)


def test_no_policy_preserves_legacy_cost_and_job_identity():
    j=job();before=copy.deepcopy(j)
    assert r.cpu_cost(j,None)==4
    assert not r.can_admit(j,None,9,12,0,2,0,0,float('inf'))
    assert j==before


def test_invalid_policy_does_not_enable_unbounded_launches(tmp_path,policy):
    policy['cpu_network_cost']=0
    (tmp_path/'resource_policy.json').write_text(json.dumps(policy))
    with pytest.raises(ValueError):r.read_policy(tmp_path)


def test_policy_is_reloaded_for_staged_rollout(tmp_path,policy):
    p=tmp_path/'resource_policy.json'
    p.write_text(json.dumps(policy));assert r.read_policy(tmp_path)['max_cpu_network_jobs']==8
    policy['max_cpu_network_jobs']=4;p.write_text(json.dumps(policy))
    assert r.read_policy(tmp_path)['max_cpu_network_jobs']==4
