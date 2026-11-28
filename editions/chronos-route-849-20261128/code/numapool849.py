"""Two native process families, each sharing an unmodified CPU-local Chronos instance."""
import multiprocessing as mp,os,queue,time,traceback
from pathlib import Path
def cpus(node):
    paths=Path('/sys/devices/system/cpu');aff=os.sched_getaffinity(0);picked=[]
    for f in sorted(paths.glob('cpu[0-9]*/topology/core_id'),key=lambda x:int(x.parents[1].name[3:])):
        parent=f.parents[1];cpu=int(parent.name[3:])
        if cpu not in aff or not (parent/f'node{node}').exists():continue
        core=(int((f.parent/'physical_package_id').read_text()),int(f.read_text()))
        if all(other!=core for _,other in picked):picked.append((cpu,core))
    return [c for c,_ in picked]
def worker(fn,inq,outq,cpu,rank,local_cpus):
    if fn.__name__ in ('labelworld','confirm_job'):
        # The preceding stage can still be finishing cache-only forward passes.
        # Admit native label jobs onto those cores as each prefetch worker exits.
        marker=Path(__file__).resolve().parents[1]/'workers'
        while True:
            active=0
            for p in marker.glob('*'):
                try:
                    if Path('/proc',p.name).exists():
                        msg=p.read_text()
                        if (fn.__name__=='labelworld' and '提前计算' in msg) or (fn.__name__=='confirm_job' and '开发闭环提前运行' in msg):active+=1
                except OSError:pass
            if rank<max(1,47-active):break
            time.sleep(2)
    import fcntl
    root=Path(__file__).resolve().parents[1];locks=root/'cache/cpu_allocation';locks.mkdir(exist_ok=True)
    pool=[cpu]+[c for c in local_cpus if c!=cpu]
    held=None
    while held is None:
        occupied=set()
        for marker in (root/'workers').glob('*'):
            try:
                affinity=os.sched_getaffinity(int(marker.name))
                if len(affinity)==1:occupied.update(affinity)
            except (ValueError,ProcessLookupError):pass
        for candidate in pool:
            if candidate in occupied:continue
            fd=(locks/str(candidate)).open('a')
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:fd.close();continue
            held=fd;os.sched_setaffinity(0,{candidate});break
        if held is None:time.sleep(1)
    while True:
        job=inq.get()
        if job is None:return
        i,arg=job
        try:outq.put((i,True,fn(arg)))
        except BaseException:outq.put((i,False,traceback.format_exc()));return
def leader(fn,inq,outq,cores):
    node=int(Path(f'/sys/devices/system/cpu/cpu{cores[0]}/node1').exists());local_cpus=cpus(node)
    os.sched_setaffinity(0,set(local_cpus))
    from base849 import pf
    pf.chrono()
    fork=mp.get_context('fork')
    ps=[fork.Process(target=worker,args=(fn,inq,outq,c,2*j+node,local_cpus)) for j,c in enumerate(cores)]
    try:
        for p in ps:p.start()
        for p in ps:p.join()
        if any(p.exitcode for p in ps):raise RuntimeError('NUMA worker exited abnormally')
    finally:
        for p in ps:
            if p.is_alive():p.terminate()
def parallel(fn,jobs,n):
    jobs=list(jobs)
    if not jobs:return []
    n=min(len(jobs),n);groups=[cpus(0),cpus(1)]
    if not all(groups):raise ValueError('both NUMA nodes must be available on this host')
    counts=[(n+1)//2,n//2];ctx=mp.get_context('spawn');inq=ctx.Queue();outq=ctx.Queue();leaders=[]
    for cores,count in zip(groups,counts):
        if count:
            p=ctx.Process(target=leader,args=(fn,inq,outq,cores[:count]));p.start();leaders.append(p)
    for i,x in enumerate(jobs):inq.put((i,x))
    for _ in range(n):inq.put(None)
    result=[None]*len(jobs);done=0
    try:
        while done<len(jobs):
            try:i,ok,value=outq.get(timeout=10)
            except queue.Empty:
                if any(p.exitcode not in (None,0) for p in leaders) or not any(p.is_alive() for p in leaders):raise RuntimeError('incomplete native process pool')
                continue
            if not ok:raise RuntimeError(value)
            result[i]=value;done+=1
            print(f'{fn.__name__} {done}/{len(jobs)}',flush=True)
        for p in leaders:p.join()
        return result
    finally:
        for p in leaders:
            if p.is_alive():
                # Terminate only this pool's subprocess tree on failure.
                kids=Path(f'/proc/{p.pid}/task/{p.pid}/children')
                if kids.exists():
                    for child in kids.read_text().split():
                        try:os.kill(int(child),15)
                        except ProcessLookupError:pass
                p.terminate();p.join()
        inq.cancel_join_thread();outq.cancel_join_thread();inq.close();outq.close()
