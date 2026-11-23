"""Start the original campaign with measured admission limits and a recovery record."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/chen')
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('migration_resume',ROOT/'migration_20261117/resume_experiments.py')
resume=importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)
c=resume.c


def main():
    report=resume.preflight()
    c.save(HERE/'startup_preflight.json',report)
    policy=c.read(HERE/'resource_policy.json')
    resume.resources.read_policy(HERE)
    if report['gpu']['gpu_memory_total_gib']<22 or report['resources']['memory_limit_bytes']<40*2**30:
        raise RuntimeError('This policy is for the measured 22.5 GiB GPU / 46 GiB RAM server.')
    if '--check-only' in sys.argv:
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return
    with (c.RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=c.read(c.RUN/'resource_policy.json')
        policy['updated_at']=c.now()
        history=dict(preflight=report,previous_policy=previous,new_policy=policy,
            probe=str(HERE/'probe_results.json'),
            scientific_source_hashes_match=c.code_hashes()==c.read(c.RUN/'source_hashes.json'),
            orchestration_hashes={str(p):c.digest(p) for p in [ROOT/'formal_execution/campaign.py',ROOT/'formal_execution/resource_control.py']},
            note='Resumed on Quadro RTX 6000; balanced ready network families, four weighted GPU slots; scientific models and plan unchanged.')
        c.save(c.RUN/'migration_history'/f'optimized_{time.time_ns()}.json',history)
        c.save(c.RUN/'resource_policy.json',policy)
    log_path=c.RUN/'migration_supervisor.log'
    with log_path.open('a') as log:
        process=subprocess.Popen([str(c.PYTHON),str(ROOT/'formal_execution/campaign.py'),'run'],
            cwd=ROOT/'formal_execution',env=c.environment(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError(f'Supervisor exited: {process.returncode}; inspect {log_path}')
    c.save(HERE/'launch.json',dict(at=c.now(),pid=process.pid,log=str(log_path)))
    print(f'Started supervisor PID={process.pid}; log={log_path}',flush=True)


if __name__=='__main__':
    main()
