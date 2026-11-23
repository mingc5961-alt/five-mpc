"""Resource-aware admission and graceful transfer of an existing worker pool."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from collections import deque


def gpu_task(row):
    return row['method'] in ('C','CR_raw')


class Admission:
    def __init__(self):
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        self.workers=max(1,min(len(os.sched_getaffinity(0)),math.ceil(int(quota)/int(period)) if quota!='max' else os.cpu_count()))
        self.gpu_slots=min(3,self.workers)
        self.samples=deque(maxlen=6)
        self.last_change=self.last_sample=time.monotonic()
        self.values={}
        self.cpu=self.cpu_ticks()

    @staticmethod
    def cpu_ticks():
        return int(Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines()[0].split()[1])/1e6

    def sample(self, gpu_count):
        now=time.monotonic()
        if now-self.last_sample<5:return
        cpu=self.cpu_ticks()
        self.values['cpu_cores']=round((cpu-self.cpu)/(now-self.last_sample),2)
        self.cpu,self.last_sample=cpu,now
        try:
            query=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits','-i','0'],text=True,timeout=4)
            util,used,total=map(int,query.strip().split(','))
        except (OSError,subprocess.SubprocessError,ValueError): return
        self.samples.append(util)
        mean=sum(self.samples)/len(self.samples)
        self.values.update(gpu_pct=util,gpu_avg_pct=round(mean,1),gpu_mib=used,gpu_total_mib=total)
        # Each measured process occupied < 1 GiB. Leave room for startup,
        # context growth and other workloads before admitting another model.
        if total-used<2048:
            self.gpu_slots=max(1,gpu_count-1)
            self.last_change=now
        elif now-self.last_change>75 and len(self.samples)==6 and gpu_count>=self.gpu_slots and mean<80 and total-used>4096:
            self.gpu_slots=min(6,self.workers,self.gpu_slots+1)
            self.last_change=now
        self.values.update(worker_limit=self.workers,gpu_limit=self.gpu_slots)

    def next_job(self,pending,gpu_count,reserved):
        def eligible(row):
            if row['out'] in reserved:return False
            if row['method']=='fixed-grid':return True
            return Path(row['argv'][row['argv'].index('--bundle')+1]).is_file()
        # Fill unoccupied GPU slots first. Each world is an independent process.
        for want_gpu in ([True,False] if gpu_count<self.gpu_slots else [False]):
            for row in pending:
                if gpu_task(row)==want_gpu and eligible(row):
                    pending.remove(row)
                    return row
        return None


class SupervisorLock:
    def __init__(self,root,old_pid=None):
        self.root=root
        self.lock=(root/'supervisor.lock').open('a')
        self.leader=(root/'admission.lock').open('a')
        fcntl.flock(self.leader,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.reserved=set()
        self.owned=False
        if self.acquire():return
        if not old_pid:raise RuntimeError('Existing supervisor holds the lock')
        proc=Path(f'/proc/{old_pid}')
        runtime=proc.joinpath('cmdline').read_bytes().split(b'\0')
        if int((root/'supervisor.pid').read_text())!=old_pid or not any(x.endswith(b'ch6_run.py') for x in runtime):
            raise RuntimeError('Supervisor identity mismatch')
        if not any(os.path.realpath(fd)==str(root/'supervisor.lock') for fd in (proc/'fd').iterdir()):
            raise RuntimeError('Supervisor lock identity mismatch')
        # Signal the coordinator PID only. ThreadPoolExecutor waits for all
        # submitted futures (including retries) during KeyboardInterrupt.
        os.kill(old_pid,signal.SIGINT)
        time.sleep(2)
        if self.acquire():return
        self.reserved.update(json.loads((root/'status.json').read_text())['running'])
        for area in ('fixed','validation','test'):
            parent=root/area
            for path in parent.glob('*/*.work.*'):
                self.reserved.add(str(path).rsplit('.work.',1)[0])
        self.reserved={p for p in self.reserved if not (Path(p)/'completed.json').exists()}
        (root/'handover.json').write_text(json.dumps(dict(old_pid=old_pid,new_pid=os.getpid(),
              retained=sorted(self.reserved),at=time.time()),indent=2))

    def acquire(self):
        if self.owned:return True
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        self.owned=True
        self.reserved.clear()
        return True

    def remaining(self):
        if not self.acquire():
            self.reserved={p for p in self.reserved if not (Path(p)/'completed.json').exists()}
        return self.reserved
