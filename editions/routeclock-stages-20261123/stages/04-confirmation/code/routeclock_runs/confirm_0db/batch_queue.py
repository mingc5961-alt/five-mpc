"""Persistent scheduler for the frozen 0 dB campaign. Run with the verified venv."""
import collections
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
STATE=ROOT/'state'; RESULTS=ROOT/'results'
TARGET=Path('/data/chen/refine-logs/实时实验进度.txt')
MODELS=('Chronos-2','VECTOR','RAFT')
WINS=json.loads((ROOT/'work/flight_windows.json').read_text())
TOTAL_FLIGHTS=15*len(WINS)
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'work/code/routeclock_refine'),OMP_NUM_THREADS='1',
         OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
CG=Path('/sys/fs/cgroup')

def existing(script):
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if (p/'cwd').resolve()==ROOT and len(args)>1 and Path(os.fsdecode(args[1])).name==script:
                yield int(p.name)
        except OSError:continue

def jobs():
    yield 'baselines',['run_baselines.py'],(),6,0
    for lo in (0,5):
        for model in MODELS:
            tag=f'dev_{model}_{lo}'
            yield tag,['queue_advanced.py',model,'development','7',str(lo),str(lo+5)],(),1,1
    for model in MODELS:
        dev=[f'dev_{model}_{lo}' for lo in (0,5)]
        selected='select_'+model
        yield selected,['queue_advanced.py','select',model],dev,1,0
    for model in ('I','RC',*MODELS):
        deps=() if model in ('I','RC') else ['select_'+model]
        yield 'flight_'+model,['run_flights.py',model],deps,1,int(bool(deps))
    for seed in (7,11,19):
        for lo in (0,20):
            for model in MODELS:
                yield f'confirm_{model}_{seed}_{lo}',['queue_advanced.py',model,'confirmation',str(seed),str(lo),str(lo+20)],['select_'+model],1,1

def read_cpu():
    return int(dict(x.split() for x in (CG/'cpu.stat').read_text().splitlines())['usage_usec'])
def publish(batch,finished,active,errors,t0,c0):
    from live_report import render
    now=time.monotonic();cc=read_cpu()
    interval=now-t0
    snapshot=render(ROOT,batch,finished,active,errors,(cc-c0)/max(.001,interval)/1e6,now,interval)
    tmp=TARGET.with_suffix('.queue.tmp');tmp.write_text(snapshot);tmp.replace(TARGET)
    return now,cc

def inherited(batch):
    import psutil
    found={}
    class Running:
        def __init__(self,proc):
            self.proc=proc
        def poll(self):
            try:
                if self.proc.is_running() and self.proc.status()!=psutil.STATUS_ZOMBIE:return None
            except psutil.NoSuchProcess:pass
            # Re-enter through the ordinary worker to validate all completion receipts.
            return 76
    lookup={tuple(v[0]):k for k,v in batch.items() if k!='baselines'}
    for proc in psutil.process_iter(['cmdline']):
        try:
            args=tuple((proc.info['cmdline'] or [])[1:])
            if args in lookup and Path(proc.cwd())==ROOT:found[lookup[args]]=Running(proc)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found

def main():
    lock=(STATE/'batch_queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    batch={tag:(cmd,deps,cpu,gpu) for tag,cmd,deps,cpu,gpu in jobs()}
    batch['summary']=(['summarize_batch.py'],tuple(batch),1,0)
    done_dir=STATE/'jobs';done_dir.mkdir(exist_ok=True)
    finished={j for j in batch if (done_dir/(j+'.done')).is_file()}
    current=inherited(batch);errors=collections.Counter();retry={}
    t,c=time.monotonic(),read_cpu()
    while True:
        for tag,proc in list(current.items()):
            rc=proc.poll()
            if rc is None:continue
            del current[tag]
            if rc==76:continue
            if rc==0:
                finished.add(tag);(done_dir/(tag+'.done')).touch()
            else:
                errors[tag]+=1;retry[tag]=time.monotonic()+min(600,15*2**min(errors[tag],5))
        legacy_base=bool(list(existing('run_baselines.py'))) and 'baselines' not in current
        if not legacy_base and 'baselines' not in current and 'baselines' not in finished and sum(1 for _ in (RESULTS/'baselines').glob('*/complete.json'))==360:
            finished.add('baselines');(done_dir/'baselines.done').touch()
        cpus=sum(batch[j][2] for j in current)+6*legacy_base
        gpus=sum(batch[j][3] for j in current)
        for tag,(cmd,deps,cpu,gpu) in batch.items():
            if tag in finished or tag in current or not all(d in finished for d in deps):continue
            if tag=='baselines' and legacy_base:continue
            if time.monotonic()<retry.get(tag,0) or cpus+cpu>10 or gpus+gpu>7:continue
            if int((CG/'memory.current').read_text())>27*2**30:continue
            with (ROOT/'logs'/('queue_'+tag+'.log')).open('a') as log:
                current[tag]=subprocess.Popen([sys.executable,*cmd],env=ENV,stdout=log,stderr=subprocess.STDOUT)
            cpus+=cpu;gpus+=gpu
        active=list(current)+(['baselines'] if legacy_base else [])
        t,c=publish(batch,finished,active,errors,t,c)
        if len(finished)==len(batch):break
        time.sleep(10)

if __name__=='__main__':main()
