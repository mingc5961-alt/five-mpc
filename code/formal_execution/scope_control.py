"""Explicit user-authorized scope amendments; the original plan stays immutable."""
import copy
import hashlib
import json
from pathlib import Path


def effective_plan(root):
    root = Path(root)
    path = root/'plan.json'
    original = json.loads(path.read_text())
    amendment_path = root/'scope_amendment.json'
    if not amendment_path.exists():
        return original
    amendment = json.loads(amendment_path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != amendment['original_plan_sha256']:
        raise ValueError('Scope amendment does not match the frozen original plan')
    deferred = set(amendment['deferred_job_ids'])
    jobs = {j['id']: j for j in original['jobs']}
    if len(deferred) != len(amendment['deferred_job_ids']) or not deferred <= jobs.keys():
        raise ValueError('Unknown or duplicate deferred job identities')
    for id in deferred:
        job = jobs[id]
        if job['kind'] != 'network' or job['meta']['block'] != 'repeat':
            raise ValueError('Only supplemental repeat networks may be deferred')
        if (root/'jobs'/id/'receipt.json').exists():
            raise ValueError('A completed or failed task cannot be hidden by scope reduction')
    result = copy.deepcopy(original)
    result['jobs'] = [j for j in result['jobs'] if j['id'] not in deferred]
    for job in result['jobs']:
        removed = set(job['dependencies']) & deferred
        if removed and job['kind'] != 'summary':
            raise ValueError('Only summary dependencies may omit deferred repeat tasks')
        job['dependencies'] = [id for id in job['dependencies'] if id not in deferred]
    count = sum(j['kind'] == 'network' for j in result['jobs'])
    if count != amendment['retained_network_jobs']:
        raise ValueError('Approved scope count does not match selected jobs')
    result['counts']['five_network_runs'] = count
    result['scope_amendment'] = amendment
    return result


def network_jobs(root, head, seed):
    return [j for j in effective_plan(root)['jobs'] if j['kind'] == 'network'
            and (j['meta']['head'],j['meta']['seed']) == (head,seed)]
