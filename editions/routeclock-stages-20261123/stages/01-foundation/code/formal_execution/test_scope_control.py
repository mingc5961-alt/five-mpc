import copy
import hashlib
import json
from unittest.mock import patch
import pytest
import campaign as c
from scope_control import effective_plan, network_jobs


def fixture(root):
    def job(id,kind='network',block='repeat',deps=()):
        return dict(id=id,kind=kind,meta=dict(block=block,head='tabpfn',seed=7),dependencies=list(deps),command=['unchanged',id])
    plan=dict(counts=dict(five_network_runs=3),jobs=[job('main',block='main'),job('keep'),job('omit'),job('summary','summary',deps=['keep','omit'])])
    c.save(root/'plan.json',plan)
    amendment=dict(original_plan_sha256=c.digest(root/'plan.json'),deferred_job_ids=['omit'],retained_network_jobs=2)
    c.save(root/'scope_amendment.json',amendment)
    return plan,amendment


def test_scope_keeps_frozen_plan_and_commands(tmp_path):
    original,_=fixture(tmp_path);before=(tmp_path/'plan.json').read_bytes();p=effective_plan(tmp_path)
    assert [j['id'] for j in p['jobs']]==['main','keep','summary']
    assert p['jobs'][-1]['dependencies']==['keep']
    assert p['jobs'][1]['command']==original['jobs'][1]['command']
    assert (tmp_path/'plan.json').read_bytes()==before
    assert len(network_jobs(tmp_path,'tabpfn',7))==2


def test_scope_cannot_hide_completed_or_failed_job(tmp_path):
    fixture(tmp_path);c.save(tmp_path/'jobs/omit/receipt.json',{'status':'failed'})
    with pytest.raises(ValueError,match='cannot be hidden'):effective_plan(tmp_path)


def test_scope_cannot_remove_primary_experiment(tmp_path):
    _,a=fixture(tmp_path);a['deferred_job_ids']=['main'];c.save(tmp_path/'scope_amendment.json',a)
    with pytest.raises(ValueError,match='supplemental'):effective_plan(tmp_path)


def test_scope_binds_original_plan_hash(tmp_path):
    p,_=fixture(tmp_path);p['counts']['five_network_runs']=4;c.save(tmp_path/'plan.json',p)
    with pytest.raises(ValueError,match='frozen'):effective_plan(tmp_path)


def test_scope_does_not_fake_completion(tmp_path):
    fixture(tmp_path);p=effective_plan(tmp_path);p.update(extension_deadline='2026-11-17T10:17:32+08:00')
    for j in p['jobs']:c.save(tmp_path/'jobs'/j['id']/'receipt.json',dict(status='passed',finished_at='2026-11-17T23:00:00+08:00'))
    with patch.object(c,'RUN',tmp_path):
        status=c.publish_status(p,{j['id']:'passed' for j in p['jobs']})
    assert status['approved_scope_complete']
    assert not status['core_complete'] and not status['extensions_eligible']
    assert status['counts']==dict(passed=3,deferred=1)
    assert status['five_network_expected']==690
    assert status['approved_five_network_expected']==2
    assert not (tmp_path/'core_completion.json').exists()
    assert (tmp_path/'approved_scope_completion.json').exists()


def test_pending_retained_work_stays_incomplete(tmp_path):
    fixture(tmp_path);p=effective_plan(tmp_path);p.update(extension_deadline='2026-11-17T10:17:32+08:00')
    with patch.object(c,'RUN',tmp_path):
        status=c.publish_status(p,{j['id']:'pending' for j in p['jobs']})
    assert not status['approved_scope_complete']
