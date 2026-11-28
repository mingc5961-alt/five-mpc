"""Backfill blocked forecast lanes with distinct scheduled endpoints using measured CPU headroom."""
import os,time,fcntl
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from design849 import *
from filelock import FileLock,Timeout
LANES=10
def cpu():
    v=dict(l.split() for l in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    return time.monotonic(),int(v['usage_usec'])*1e-6,int(v['nr_throttled']),int(v['nr_periods'])
def jobkey(j):
    phase,_,k,r,s,m=j;g='dev2' if phase=='development' else 'c2' if m==.75 else 'extra';return f'{g}_{k}_{r}_{s}_{m}'
def live(key):
    p=OUT/'jobs'/(key+'.json')
    if not p.is_file():return False
    d=get(p)
    if d.get('receipt'):return Path(d['receipt']).exists()
    pid=d.get('pid')
    try:return bool(pid and Path(f'/proc/{pid}/stat').read_text().split()[21]==d['start'])
    except OSError:return False
def work(arg):
    lane,j=arg;slotdir=OUT/'cache/cpu_allocation';slotdir.mkdir(exist_ok=True)
    # Use high-numbered physical cores on either socket. The same locks are
    # honored by the main pools; no environment or native model changes are needed.
    choices=[c for pair in zip(range(47,23,-1),range(95,71,-1)) for c in pair]
    held=None
    for c in choices[lane:]+choices[:lane]:
        fd=(slotdir/str(c)).open('a')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:fd.close();continue
        held=fd;os.sched_setaffinity(0,{c});break
    if held is None:return False
    try:
        key=jobkey(j);lockdir=OUT/'jobs/locks';lockdir.mkdir(exist_ok=True)
        try:
            with FileLock(str(lockdir/(key+'.lock')),timeout=0):
                if live(key):return False
                os.environ['CHRONOS849_OVERFLOW_LANE']=str(lane)
                from sharedcache849 import SharedPrefixCache
                mark(key);stem=endpoint(j,cache_factory=SharedPrefixCache);finish(key,OUT/f'E2/receipts/{j[0]}/{stem}.json');return True
        except Timeout:return False
    finally:held.close()
def main():
    secondary='--supplement' in sys.argv
    limit=16 if secondary else LANES
    prefix='surplus' if secondary else 'overflow'
    permit_name=prefix+'_enabled.txt'
    os.environ['CHRONOS849_OVERFLOW_PERMIT']=permit_name
    w,_=world('development',1400);Model(w,'Chronos')
    dev=[('development','development',k,r,s,m) for m in (1.,.45,.60,.75) for r,s in reversed(MATRIX) for k in (1403,1402,1407,1405,1406,1404,1401,1400)]
    allowed=0;before=cpu();pending={};frozen=None;confirm=[];enabled=OUT/'logs'/permit_name;enabled.write_text('0')
    with ProcessPoolExecutor(limit,mp_context=mp.get_context('fork')) as pool:
        while not (OUT/'report_receipt.json').exists():
            time.sleep(5)
            after=cpu();usage=(after[1]-before[1])/(after[0]-before[0]);throttle=(after[2]-before[2])/max(1,after[3]-before[3]);before=after
            for lane,(future,j) in list(pending.items()):
                if future.done():future.result();pending.pop(lane)
            base_status=OUT/'logs/overflow_status.json'
            eligible=not secondary or pending or (base_status.exists() and get(base_status)['assigned']>=9)
            if usage<45.4 and eligible:allowed=min(limit,allowed+max(1,int(47.-usage)))
            elif usage>47.5 and throttle>.20:allowed=max(0,allowed-1)
            atom(enabled,str(allowed).encode())
            if (OUT/'audits/final-smoke.json').exists():
                if frozen is None:
                    frozen=get(OUT/'design/frozen.json');check(frozen)
                    # Work the common scale on higher IDs ahead of the main wave.
                    pairs=sorted(frozen['base_endpoints'],key=lambda j:(-j[0],j[1],j[2]))+frozen['extra_endpoints']
                    confirm=[('confirmation','confirmation',k,r,s,m) for k,r,s,m in pairs]
                jobs=confirm
            elif (OUT/'design/frozen.json').exists():jobs=[]
            elif (OUT/'E2/labels/VALUE-route.json').exists():jobs=dev
            else:jobs=[]
            if secondary:jobs=jobs[len(jobs)//2:]+jobs[:len(jobs)//2]
            underway={j for _,j in pending.values()}
            for lane in range(allowed):
                if lane in pending:continue
                for j in jobs:
                    if j in underway or live(jobkey(j)):continue
                    pending[lane]=(pool.submit(work,(lane,j)),j);underway.add(j);break
            put(OUT/'logs'/(prefix+'_status.json'),dict(cpu=round(usage,2),throttle_ratio=round(throttle,2),available=allowed,assigned=len(pending),jobs=[j for _,j in pending.values()]))
if __name__=='__main__':main()
