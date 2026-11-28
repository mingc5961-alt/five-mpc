"""Keep this project's scientific jobs and progress publication alive through the requested date."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from history import ROOT,put

TZ=ZoneInfo('Asia/Shanghai')
END=dt.datetime(2026,11,29,10,30,tzinfo=TZ)
PY='/data/chen/server_assessment/venv/bin/python'
def live(cmd):
    candidates=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if cmd.encode() in args and (p/'cwd').resolve()==ROOT:
                pid=int(p.name)
                if os.getsid(pid)==pid:return pid
                candidates.append(pid)
        except (OSError,RuntimeError):pass
    return min(candidates,default=None)

def start(script,logname,*args):
    with (ROOT/'logs'/logname).open('ab') as log:
        return subprocess.Popen([PY,'-Bu',script,*args],cwd=ROOT,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True).pid

def load(path,default):
    try:return json.loads(path.read_bytes())
    except (OSError,ValueError):return default

def main():
    lock=(ROOT/'accounting/watch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tracking=ROOT/'accounting/watch_attempts.json'
    history=load(tracking,{})
    deadline=ROOT/'accounting/watch_deadline.json'
    if not deadline.exists():put(deadline,dict(deadline_local=END.isoformat(),minimum_supervision=True))
    last_probe=0
    previous_inputs=None
    cpu_sample=None
    last_minute=None
    idle_since={}
    peak=load(ROOT/'accounting/watch.json',{}).get('peak_cpu_cores',0.)
    while True:
        now=dt.datetime.now(TZ)
        end=dt.datetime.fromisoformat(load(deadline,{'deadline_local':END.isoformat()})['deadline_local'])
        jobs=[
            ('independent/paired_grid.py','paired_linux_v2/summary.json','logs/paired_grid.log','paired_linux_v1/worlds/index.json'),
            ('independent/native_matrix.py','paired_linux_v2/native_matrix/summary.json','logs/paired_native.log','paired_linux_v2/summary.json'),
        ]
        extra=load(ROOT/'accounting/watch_jobs.json',[])
        jobs.extend(tuple(s) for s in extra)
        statuses={}
        publisher=live('progress.py')
        if not publisher:
            publisher=start('progress.py','progress_server.log','--serve','18765','--publish')
        for filename in ('progress_publisher.pid','progress_server.pid'):
            path=ROOT/'logs'/filename
            if not path.exists() or path.read_text().strip()!=str(publisher):
                path.write_text(str(publisher)+'\n')
        if now.timestamp()-last_probe>=60 and not (ROOT/'recovery_01/reference_input/summary.json').exists():
            bases=(Path('/data/chen'),Path('/data/chen/refine-logs'))
            candidates=[p for b in bases for p in (b/'confirm0.world',b/'factorial_run/confirm0.world') if p.is_file()]
            candidates.extend(p for b in bases for p in b.glob('*.zip') if any(s in p.name.lower() for s in ('g1','world','physics')))
            current_inputs=[]
            for path in candidates:
                try:
                    info=path.stat()
                except FileNotFoundError:
                    continue
                current_inputs.append((str(path),info.st_size,info.st_mtime_ns))
            current_inputs.sort()
            retry=(ROOT/'recovery_01/reference_input/confirm0.world').exists() and int(now.timestamp()//60)%15==0
            if (current_inputs!=previous_inputs or retry) and not live('independent/reference_rescue.py'):
                start('independent/reference_rescue.py','reference_rescue.log')
                previous_inputs=current_inputs
            last_probe=now.timestamp()
        for script,final,log,prereq,*release in jobs:
            if (ROOT/final).is_file():statuses[script]='完成';continue
            if release and now<dt.datetime.fromisoformat(release[0]):
                statuses[script]='定时任务 '+release[0];continue
            deps=[prereq] if isinstance(prereq,str) else prereq
            if not all((ROOT/path).is_file() for path in deps):statuses[script]='等前置';continue
            proc=live(script)
            if proc:
                statuses[script]=f'运行 PID {proc}'
                # A parent can be waiting for healthy children. Sum this job's own processes.
                cpu_ticks=0
                for entry in Path('/proc').glob('[0-9]*'):
                    try:
                        if script.encode() not in (entry/'cmdline').read_bytes().split(b'\0') or (entry/'cwd').resolve()!=ROOT:continue
                        fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
                        cpu_ticks+=int(fields[11])+int(fields[12])
                    except (OSError,ValueError,IndexError):pass
                then,old_pid,previous_ticks=idle_since.setdefault(script,(now.timestamp(),proc,cpu_ticks))
                if proc!=old_pid or abs(cpu_ticks-previous_ticks)>100:
                    idle_since[script]=(now.timestamp(),proc,cpu_ticks)
                elif now.timestamp()-then>1800:
                    statuses[script]=f'运行 PID {proc}，正在核对长时间空闲'
                    waiting_model=script=='independent/learned_matrix.py' and not (ROOT/'paired_linux_v2/learning/summary.json').exists()
                    memory=int(Path('/sys/fs/cgroup/memory.current').read_text())
                    if (not waiting_model and now.timestamp()-then>3600 and memory<80*2**30
                            and shutil.disk_usage(ROOT).free>20*2**30 and os.getsid(proc)==proc and os.getpgid(proc)==proc):
                        with (ROOT/'logs/watch_recoveries.jsonl').open('a') as f:
                            f.write(json.dumps(dict(time=now.isoformat(),script=script,pid=proc,reason='no CPU progress for 60 minutes'))+'\n')
                        os.killpg(proc,signal.SIGTERM)
                        idle_since.pop(script,None)
                continue
            previous=history.get(script,dict(starts=[]))
            recent=[v for v in previous['starts'] if now.timestamp()-v<3600]
            if recent and now.timestamp()-recent[-1]<min(900,30*2**min(len(recent),5)):
                statuses[script]='重试退避，详见日志';continue
            proc=start(script,Path(log).name)
            recent.append(now.timestamp());history[script]=dict(starts=recent)
            put(tracking,history);statuses[script]=f'已启动 PID {proc}'
        active=any('PID' in v for v in statuses.values())
        sample=(time.monotonic(),int(Path('/sys/fs/cgroup/cpu.stat').read_text().split()[1]))
        cores=None if cpu_sample is None else round((sample[1]-cpu_sample[1])/(sample[0]-cpu_sample[0])/1e6,1)
        cpu_sample=sample
        if cores is not None:peak=max(cores,peak)
        put(ROOT/'accounting/watch.json',dict(pid=os.getpid(),heartbeat_local=now.isoformat(timespec='seconds'),
             deadline_local=end.isoformat(timespec='minutes'),service='s6',jobs=statuses,calculation_active=active,
             cpu_cores=cores,peak_cpu_cores=peak,deadline_reached=now>=end))
        minute=now.strftime('%Y-%m-%d %H:%M')
        if minute!=last_minute:
            with (ROOT/'accounting/watch_history.jsonl').open('a') as log:
                log.write(json.dumps(dict(time=now.isoformat(timespec='seconds'),cpu_cores=cores,
                    memory_GiB=round(int(Path('/sys/fs/cgroup/memory.current').read_text())/2**30,2),
                    disk_free_GiB=round(shutil.disk_usage(ROOT).free/2**30,2),jobs=statuses),ensure_ascii=False)+'\n')
            last_minute=minute
        if now>=end and not (ROOT/'accounting/上午值守回执.txt').exists():
            txt=Path('/data/chen/refine-logs/G1原版正式实验实时进度.txt')
            if txt.is_file():shutil.copyfile(txt,ROOT/'accounting/上午值守回执.txt')
        if now>=end and all(status=='完成' for status in statuses.values()):
            # Keep a usable read-only TXT after the target time. s6's down flag preserves this completion on restarts.
            service=ROOT/'service/g1-experiment'
            (service/'down').touch()
            sv=Path('/etc/services.d/g1-experiment')
            if sv.is_dir():subprocess.run(['/command/s6-svc','-d',str(sv)],check=False)
            return
        time.sleep(15)
if __name__=='__main__':main()
