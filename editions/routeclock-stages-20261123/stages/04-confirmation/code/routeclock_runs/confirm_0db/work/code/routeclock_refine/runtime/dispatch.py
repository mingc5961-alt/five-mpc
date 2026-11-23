"""Resume frozen E22 cells and share unfinished worlds among resident workers."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from refine import campaign, network
from refine.data import BASE, ROOT
from runtime.fast_ledger import DeferredLedger

RUNTIME = BASE / 'runtime'
CONFIRM = BASE / 'tube_network' / 'confirmation'
TUBES = ('native', 'lightgbm', 'fixed')
SHARDS = [(name, first) for first in (0,35) for name in campaign.MODELS]

def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
    except FileNotFoundError:
        return []

def shard_args(name, first):
    return [b'-m', b'refine.campaign', b'worker', name.encode(), str(first).encode(), str(first+35).encode()]

def prior_workers(pid):
    status = json.loads((BASE / 'tube_campaign_status.json').read_text())
    if status['stage'] != 'freeze_and_confirmation' or status['pid'] != pid or command(pid)[1:] != [b'-m', b'refine.campaign']:
        raise RuntimeError('original campaign identity or stage changed')
    workers = {}
    for proc in Path('/proc').glob('[0-9]*'):
        args = command(proc.name)[1:]
        for name, first in SHARDS:
            if args == shard_args(name,first):
                parent = (proc/'stat').read_text().rsplit(')',1)[1].split()[1]
                if int(parent) != pid:
                    raise RuntimeError('unexpected owner of original shard')
                workers[name,first] = int(proc.name)
    if set(workers) != set(SHARDS):
        raise RuntimeError('original shard set changed')
    return workers

def log_for(name, first):
    return BASE / 'logs' / f'confirm_{name}_{first}.log'

def disk_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ensure_validated():
    proofs = {}
    code_sha = disk_sha(ROOT / 'runtime' / 'fast_ledger.py')
    for model in campaign.MODELS:
        path = RUNTIME / 'validation_replay' / (model + '_verified.json')
        proof = json.loads(path.read_text())
        if proof['sha256'] != code_sha or {c['tube'] for c in proof['checks']} != {'lightgbm','native'}:
            raise RuntimeError(f'missing validation for {model}')
        proofs[model] = disk_sha(path)
    campaign.freeze()
    spec = dict(ledger_sha256=code_sha,validation_sha256=proofs,dispatch_sha256=disk_sha(Path(__file__)),
                freeze_sha256=disk_sha(BASE/'e22_freeze.json'))
    (RUNTIME/'dispatch_manifest.json').write_text(json.dumps(spec,indent=2))

def worker(home, start, ready):
    network.Ledger = DeferredLedger
    factories = dict(network.MODELS)
    loaded = set()
    order = [(home,start)] + [job for job in SHARDS if job != (home,start)]
    import torch
    marker = RUNTIME/'active'/f'{os.getpid()}.json'
    original = network.run_world
    def tracked(cfg,world,adapter,bundle,tube,trace=False):
        part=marker.with_suffix('.part')
        part.write_text(json.dumps(dict(pid=os.getpid(),model=model,world=world.mission,
                                        phase='confirmation',tube=tube,started=time.time())))
        part.replace(marker)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            begin = time.process_time()
            result = original(cfg,world,adapter,bundle,tube,trace)
            stats = dict(cpu_seconds=time.process_time()-begin, pid=os.getpid(),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_available() else 0)
            dest=RUNTIME/'cell_resources'/model/f'{world.mission:03}'/(tube+'.json')
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix('.part')
            tmp.write_text(json.dumps(stats))
            tmp.replace(dest)
            return result
        finally:
            marker.unlink(missing_ok=True)
    network.run_world = tracked
    for model, first in order:
        if (model,first) != (home,start):
            while not ready.exists():
                time.sleep(1)
        for i in range(first,first+35):
            folder = CONFIRM / model / f'{i:03}'
            if all((folder/(tube+'.json')).exists() for tube in TUBES):
                continue
            with (RUNTIME/'world_locks'/f'{model}_{i}.lock').open('a') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if model not in loaded:
                    resident = factories[model]()
                    network.MODELS[model] = lambda obj=resident: obj
                    loaded.add(model)
                network.execute(model,'confirmation',[i],TUBES,BASE/'tube_network')

def main(pid):
    with (RUNTIME/'dispatch.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_validated()
        old = prior_workers(pid) if pid else {}
        marks = {job:log_for(*job).stat().st_size for job in old}
        ready = RUNTIME / f'handoff_{os.getpid()}.done'
        (RUNTIME/'world_locks').mkdir(exist_ok=True)
        (RUNTIME/'active').mkdir(exist_ok=True)
        lock = (BASE/'e22_campaign.lock').open('a')
        if pid:
            os.kill(pid,signal.SIGTERM)
            fcntl.flock(lock,fcntl.LOCK_EX)
        else:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any(command(p.name)[1:3] == [b'-m',b'refine.campaign'] for p in Path('/proc').glob('[0-9]*')):
                raise RuntimeError('old worker still running')
        campaign.state('freeze_and_confirmation')
        try:
            def drain(job):
                if job not in old:
                    return
                child = old[job]
                path = log_for(*job)
                expected = shard_args(*job)
                while command(child)[1:] == expected and path.stat().st_size == marks[job]:
                    time.sleep(.15)
                if command(child)[1:] == expected:
                    os.kill(child,signal.SIGTERM)
                while command(child)[1:] == expected:
                    time.sleep(.05)
                print('checkpoint handoff',job,flush=True)

            def launch(job):
                name,first = job
                with log_for(*job).open('a') as log:
                    args=[sys.executable,'-m','runtime.dispatch','--worker',name,str(first),str(ready)]
                    for attempt in range(2):
                        code=subprocess.call(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        if code==0:
                            return
                    raise RuntimeError(f'worker {job} exited with code {code}')

            with concurrent.futures.ThreadPoolExecutor(len(SHARDS)) as pool:
                drains={pool.submit(drain,j):j for j in SHARDS}
                runs=[]
                for future in concurrent.futures.as_completed(drains):
                    future.result()
                    runs.append(pool.submit(launch,drains[future]))
                ready.touch()
                for future in concurrent.futures.as_completed(runs):
                    future.result()
            campaign.results()
            campaign.state('E22-LGB_complete')
        except Exception as exc:
            campaign.state('error: '+repr(exc))
            raise

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--handover',type=int)
    p.add_argument('--worker',nargs=3)
    a=p.parse_args()
    if a.worker:
        worker(a.worker[0],int(a.worker[1]),Path(a.worker[2]))
    else:
        main(a.handover)
