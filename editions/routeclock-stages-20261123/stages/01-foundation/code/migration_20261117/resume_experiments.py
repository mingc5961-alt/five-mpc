"""Validate the saved campaign and detach its existing resumable supervisor."""
import argparse
import collections
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/data/chen')
KIT = ROOT/'migration_20261117'
sys.path.insert(0, str(ROOT/'formal_execution'))
import campaign as c
import resource_control as resources


def preflight():
    import numpy as np
    import psutil
    import torch
    if Path(__file__).resolve().parent != KIT:
        raise RuntimeError('Restore this snapshot under /data/chen; frozen commands use this absolute path.')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 required for the saved model environment.')
    if Path(sys.executable).absolute() != c.PYTHON:
        raise RuntimeError(f'Run this script with {c.PYTHON}')
    expected = json.loads((KIT/'environment.json').read_text())
    for name, version in expected['critical_packages'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'Package version mismatch: {name}, expected {version}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; check the NVIDIA driver and CUDA PyTorch installation.')
    # Execute a real CUDA kernel so incompatible drivers are found before launch.
    torch.ones((16, 16), device='cuda').matmul(torch.ones((16, 16), device='cuda'))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if c.code_hashes() != c.read(c.RUN/'source_hashes.json'):
        raise RuntimeError('Original scientific source fingerprint mismatch.')
    if c.digest(c.RUN/'plan.json') != expected['plan_sha256']:
        raise RuntimeError('Frozen experiment plan changed.')
    plan = c.read(c.RUN/'plan.json')
    for job in plan['jobs']:
        c.effective_command(job)
    state = c.states(plan)
    problem = {k: v for k, v in state.items() if v not in ('pending', 'passed')}
    if problem:
        raise RuntimeError(f'Inspect these jobs before resuming; no automatic retries: {problem}')
    cache_dir = c.RUN/'five/tabpfn_7'
    binding = c.read(cache_dir/'eval_cache_binding.json')
    if binding['bundle_sha256'] != c.digest(cache_dir/'bundle.pkl'):
        raise RuntimeError('Saved TabPFN bundle changed.')
    actual_tests = {p.name: c.digest(p) for p in sorted((c.RUN/'five_data/test').glob('*.pkl'))}
    if actual_tests != binding['test_files']:
        raise RuntimeError('Test data do not match the saved prediction cache.')
    cache_counts = {}
    for name in ('I', 'P', 'RC', 'C', 'CR_raw', 'full', 'duration'):
        with np.load(cache_dir/'inference_cache'/f'{name}.npz', allow_pickle=False) as cache:
            values = cache['values']
            suffix = (9,) if name == 'duration' else (21, 2)
            if values.shape[1:] != suffix or not 0 < len(values) <= 1200 or not np.isfinite(values).all():
                raise RuntimeError(f'Invalid saved prediction cache: {name}')
            cache_counts[name] = len(values)
    current = resources.sample()
    gpu = resources.sample_gpu()
    policy = resources.read_policy(c.RUN)
    resources.memory_budget(c.RUN, [], policy, current)
    if current['memory_limit_bytes'] < 24*2**30:
        raise RuntimeError('This recovery configuration requires at least 24 GiB RAM.')
    if gpu['gpu_memory_total_gib'] < 10:
        raise RuntimeError('At least 10 GiB GPU memory required; 24 GiB recommended for this migration.')
    if shutil.disk_usage(ROOT).free < 12*2**30:
        raise RuntimeError('Keep at least 12 GiB free for the remaining results after extraction.')
    return dict(at=c.now(), gpu_name=torch.cuda.get_device_name(0),
                python=sys.version, critical_packages=expected['critical_packages'],
                cpu_affinity=len(psutil.Process().cpu_affinity()), resources=current, gpu=gpu,
                job_counts=dict(collections.Counter(state.values())), cache_counts=cache_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    report = preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        print('Preflight passed; no experiments started.')
        return
    # Existing supervisor also takes this lock. Hold it while preparing policy.
    import fcntl
    with (c.RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_policy = resources.read_policy(c.RUN)
        policy = dict(old_policy)
        policy['cpu_capacity'] = min(12, report['cpu_affinity'], report['resources']['cpu_quota_cores'] or 12)
        if report['gpu']['gpu_memory_total_gib'] >= 22:
            policy['gpu_capacity'] = 6
            policy['gpu_memory_ceiling_gib'] = min(20, report['gpu']['gpu_memory_total_gib']-3)
        policy['updated_at'] = c.now()
        policy['scope'] = 'Migration admission settings only; model, seeds, worlds and query sizes remain frozen.'
        stamp = str(time.time_ns())
        c.save(c.RUN/'migration_history'/f'{stamp}.json', dict(
            preflight=report, previous_policy=old_policy, new_policy=policy,
            note='Completed results and caches retained. GPU hardware may differ from the original RTX 3060.'))
        c.save(c.RUN/'resource_policy.json', policy)
    log_path = c.RUN/'migration_supervisor.log'
    with log_path.open('a') as log:
        process = subprocess.Popen([str(c.PYTHON), str(ROOT/'formal_execution/campaign.py'), 'run'],
            cwd=ROOT/'formal_execution', env=c.environment(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError(f'Supervisor exited with {process.returncode}; inspect {log_path}')
    print(f'Supervisor started: PID={process.pid}; log={log_path}')
    print(f'Status: {c.RUN}/执行状态与问题.txt')
    print(f'Final report: {c.RUN}/comparison/实验结果与分析.txt')


if __name__ == '__main__':
    main()
