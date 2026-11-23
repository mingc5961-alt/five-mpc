import copy
import json
from types import SimpleNamespace

import pytest

import resource_control as r


@pytest.mark.parametrize('memory_limit,expected_limit,expected_used', [
    ('max', 64*r.GIB, 20*r.GIB),
    (str(32*r.GIB), 32*r.GIB, 8*r.GIB),
])
def test_memory_sampling_on_migrated_server(tmp_path,monkeypatch,memory_limit,expected_limit,expected_used):
    for name,value in {'cpu.max':'1300000 100000', 'memory.max':memory_limit,
                       'memory.current':str(8*r.GIB), 'cpu.stat':'usage_usec 100\n',
                       'memory.events':'oom_kill 0\n'}.items():
        (tmp_path/name).write_text(value)
    monkeypatch.setattr(r.psutil,'virtual_memory',lambda:SimpleNamespace(total=64*r.GIB,available=44*r.GIB))
    sample=r.sample(tmp_path)
    assert sample['memory_limit_bytes']==expected_limit
    assert sample['memory_current_bytes']==expected_used
    assert sample['cpu_quota_cores']==13


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


def test_gpu_networks_can_share_but_training_keeps_larger_reservation(policy):
    policy.update(tabpfn_network_gpu_slots=1, tabpfn_train_vram_gib=6.5,
                  tabpfn_network_vram_gib=3, chronos_vram_gib=1.5,
                  tabpfn_network_memory_gib=3)
    network=job(gpu=2,head='tabpfn')
    training=dict(network,kind='train')
    assert r.gpu_cost(network,policy)==1
    assert r.gpu_cost(training,policy)==2
    assert r.vram_reservation(network,policy)==3
    assert r.vram_reservation(training,policy)==6.5
    assert r.memory_reservation(network,policy)==3*r.GIB
    assert r.can_admit(network,policy,4,12,2,3,0,10*r.GIB,26*r.GIB)
    assert not r.can_admit(training,policy,4,12,2,3,0,10*r.GIB,26*r.GIB)
