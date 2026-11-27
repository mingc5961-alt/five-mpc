import time,os,signal,sys
from pathlib import Path
# Retire old executor processes once they return to their queue and stop using CPU.
pids=set(map(int,sys.argv[1:]));ticks={};quiet={}
while pids:
    for pid in list(pids):
        try:
            p=Path(f'/proc/{pid}')
            if p.resolve() is None or (p/'cwd').resolve()!=Path(__file__).resolve().parent:
                pids.remove(pid);continue
            st=(p/'stat').read_text().split();now=int(st[13])+int(st[14])
            if int(st[3])!=1 or st[2]=='Z':pids.remove(pid);continue
            wait=(p/'wchan').read_text().strip()
            idle=now==ticks.get(pid) and st[2]=='S' and ('pipe' in wait or 'futex' in wait or 'sem' in wait)
            quiet[pid]=quiet.get(pid,0)+1 if idle else 0;ticks[pid]=now
            if quiet[pid]>=5:os.kill(pid,signal.SIGTERM);pids.remove(pid)
        except (FileNotFoundError,ProcessLookupError):pids.remove(pid)
    time.sleep(5)
