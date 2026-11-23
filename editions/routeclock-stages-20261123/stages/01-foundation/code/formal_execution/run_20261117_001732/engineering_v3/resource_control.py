"""Admission control only: never changes scientific commands or model settings."""
import json
import math
import time
from pathlib import Path

GIB = 2**30
CGROUP = Path('/sys/fs/cgroup')


def read_policy(run):
    path = Path(run)/'resource_policy.json'
    if not path.exists():
        return None
    policy = json.loads(path.read_text())
    positive = ('cpu_capacity', 'cpu_network_cost', 'gpu_network_cpu_cost',
                'tabpfn_train_cpu_cost', 'memory_headroom_gib',
                'cpu_network_memory_gib', 'gpu_network_memory_gib',
                'tabpfn_memory_gib', 'other_memory_gib')
    for key in positive:
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid resource policy: {key}')
    cap = policy['max_cpu_network_jobs']
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError('Invalid max_cpu_network_jobs')
    return policy


def cpu_cost(job, policy):
    if policy is None:
        return job['cpu']
    if job['kind'] == 'network':
        return policy['gpu_network_cpu_cost' if job['gpu'] else 'cpu_network_cost']
    if job['kind'] == 'train' and job['meta']['head'] == 'tabpfn':
        return policy['tabpfn_train_cpu_cost']
    return job['cpu']


def memory_reservation(job, policy):
    if job['meta'].get('head') == 'tabpfn':
        key = 'tabpfn_memory_gib'
    elif job['kind'] == 'network':
        key = 'gpu_network_memory_gib' if job['gpu'] else 'cpu_network_memory_gib'
    else:
        key = 'other_memory_gib'
    return policy[key]*GIB


def sample(cgroup=CGROUP):
    quota, period = (cgroup/'cpu.max').read_text().split()
    memory_limit = (cgroup/'memory.max').read_text().strip()
    stats = dict(line.split() for line in (cgroup/'cpu.stat').read_text().splitlines())
    events = dict(line.split() for line in (cgroup/'memory.events').read_text().splitlines())
    return dict(monotonic=time.monotonic(),
                cpu_quota_cores=None if quota == 'max' else int(quota)/int(period),
                cpu_usage_usec=int(stats['usage_usec']),
                cpu_throttled_usec=int(stats.get('throttled_usec', 0)),
                memory_current_bytes=int((cgroup/'memory.current').read_text()),
                memory_limit_bytes=None if memory_limit == 'max' else int(memory_limit),
                oom_kill=int(events.get('oom_kill', 0)))


def sample_rates(current, previous):
    if previous is None or current['monotonic'] <= previous['monotonic']:
        return {}
    seconds = current['monotonic']-previous['monotonic']
    return dict(cpu_used_cores=round((current['cpu_usage_usec']-previous['cpu_usage_usec'])/1e6/seconds, 3),
                cpu_throttled_seconds_delta=round((current['cpu_throttled_usec']-previous['cpu_throttled_usec'])/1e6, 3))


def memory_budget(run, running, policy, current):
    """Reserve unobserved worker growth, including just-started workers."""
    limit = current['memory_limit_bytes']
    if limit is None:
        raise ValueError('Explicit cgroup memory limit required for tuned admission')
    ceiling = limit-policy['memory_headroom_gib']*GIB
    if ceiling <= 0:
        raise ValueError('Memory headroom exceeds container limit')
    reserved_growth = 0
    for job in running:
        path = Path(run)/'jobs'/job['id']/'resource.json'
        observed = json.loads(path.read_text())['rss_bytes'] if path.exists() else 0
        reserved_growth += max(0, memory_reservation(job, policy)-observed)
    return ceiling, current['memory_current_bytes']+reserved_growth


def can_admit(job, policy, cpu, cpu_capacity, gpu, gpu_capacity,
              cpu_networks, memory_committed, memory_ceiling):
    if cpu+cpu_cost(job, policy) > cpu_capacity+1e-9 or gpu+job['gpu'] > gpu_capacity:
        return False
    if policy is not None:
        if job['kind'] == 'network' and not job['gpu'] and cpu_networks >= policy['max_cpu_network_jobs']:
            return False
        if memory_committed+memory_reservation(job, policy) > memory_ceiling:
            return False
    return True
