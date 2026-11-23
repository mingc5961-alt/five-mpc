import json
from pathlib import Path
from unittest.mock import patch

import pytest

import campaign as c
import audit

def test_declared_complete_matrix_and_dependencies():
    jobs=c.build_jobs();seen=set()
    for j in jobs:
        assert j['id'] not in seen
        assert set(j['dependencies'])<=seen
        assert 0<j['cpu']<=12 and 0<=j['gpu']<=2
        seen.add(j['id'])
    data=[j for j in jobs if j['kind']=='data']
    assert sum(j['meta']['stop']-j['meta']['start'] for j in data)==2400
    assert sum(j['kind']=='network' for j in jobs)==690
    assert sum(j['kind']=='train' for j in jobs)==4
    for j in jobs:
        if j['kind']=='network' and j['meta']['head']=='tabpfn': assert j['gpu']==2

def test_failure_blocks_descendants_but_not_independent_jobs(tmp_path):
    plan={'jobs':[{'id':'a','dependencies':[]},{'id':'b','dependencies':['a']},
                  {'id':'c','dependencies':['b']},{'id':'d','dependencies':[]}]}
    with patch.object(c,'RUN',tmp_path):
        c.save(tmp_path/'jobs/a/receipt.json',{'status':'failed'})
        assert c.states(plan)==dict(a='failed',b='blocked',c='blocked',d='pending')

def test_completed_job_is_not_requeued(tmp_path):
    with patch.object(c,'RUN',tmp_path):
        c.save(tmp_path/'jobs/a/receipt.json',{'status':'passed'})
        assert c.states({'jobs':[{'id':'a','dependencies':[]}]})=={'a':'passed'}

def test_stale_worker_does_not_count_as_success(tmp_path):
    with patch.object(c,'RUN',tmp_path):
        c.save(tmp_path/'jobs/a/running.json',{'pid':999999999,'create_time':0})
        assert c.states({'jobs':[{'id':'a','dependencies':[]}]})=={'a':'interrupted'}

def test_atomic_json_and_nonfinite_values(tmp_path):
    p=tmp_path/'state.json';c.save(p,{'n':1});assert json.loads(p.read_text())=={'n':1}
    with pytest.raises(ValueError): c.save(p,{'n':float('nan')})
    assert json.loads(p.read_text())=={'n':1}

def test_incomplete_network_output_cannot_pass(tmp_path):
    folder=tmp_path/'five/network/lightgbm_7/main/003';folder.mkdir(parents=True)
    (folder/'completed.json').write_text('{}')
    with pytest.raises((FileNotFoundError,KeyError,ValueError)):
        audit.verify_job({'kind':'network','meta':dict(head='lightgbm',seed=7,block='main',index=3,mission=0,method='RC',mode='joint',reuse=None)},tmp_path)

def test_bootstrap_does_not_suppress_negative_results():
    mean,low,high=audit.bootstrap([1,2,3],[4,5,6])
    assert (mean,low,high)==(-3.,-3.,-3.)

def test_ten_hour_condition_uses_completion_time_not_poll_time():
    deadline='2026-11-17T10:17:32+08:00'
    assert c.extension_gate(['2026-11-17T10:17:31+08:00'],1,deadline)[1]
    assert c.extension_gate([deadline],1,deadline)[1]
    assert not c.extension_gate(['2026-11-17T10:17:33+08:00'],1,deadline)[1]
    assert not c.extension_gate(['2026-11-17T09:00:00+08:00'],2,deadline)[1]

def test_partial_report_exposes_failure_without_claiming_completion(tmp_path):
    c.save(tmp_path/'plan.json',{'jobs':[{'id':'bad'}]})
    c.save(tmp_path/'status.json',{'counts':{'failed':1}})
    c.save(tmp_path/'jobs/bad/receipt.json',{'job':'bad','status':'failed','error':'Out of memory'})
    audit.partial_report(tmp_path)
    assert not c.read(tmp_path/'comparison/completion_audit.json')['complete']
    assert 'Out of memory' in (tmp_path/'comparison/未完成实验的问题分析.txt').read_text()
    assert (tmp_path/'comparison/partial_network_jobs.csv').exists()
