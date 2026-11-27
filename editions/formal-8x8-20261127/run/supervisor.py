import fcntl,sqlite3,concurrent.futures as cf,multiprocessing as mp,signal,sys
from base import *
from live import publish
from worker import run,init,CUDA_HOST_CPUS
from panel_reuse import PANEL_SOURCE
POOL_LIMITS = {'cpu': 11, 'gpu': 2, 'light': 1}

def proc_alive(pid):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def main():
    lock=(BASE/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads((BASE/'manifest.json').read_text());tasks.sort(key=lambda t:(t['prio'],t['id']))
    assert len(tasks)==len({t['id'] for t in tasks})
    all_ids={t['id'] for t in tasks};assert all(set(t['deps'])<=all_ids for t in tasks)
    panel_sources = {}
    for task in tasks:
        a = task['args']
        if task['kind'] == 'panel' and a['mode'] == 'all' and a['a'] in PANEL_SOURCE:
            key = 'panel__' + tag(dict(a, a=PANEL_SOURCE[a['a']]))
            if key in all_ids:
                panel_sources[task['id']] = key
    db=sqlite3.connect(BASE/'queue.sqlite3');db.execute('pragma journal_mode=WAL')
    db.execute('create table if not exists jobs(id text primary key,status text,attempts integer,note text,updated real)')
    db.executemany("insert or ignore into jobs values(?,'pending',0,'',?)",[(t['id'],time.time()) for t in tasks]);db.commit()
    db.execute("create temp table selected(id text primary key)")
    db.executemany("insert into selected values(?)", [(key,) for key in all_ids])
    db.execute("update jobs set status='cancelled' where id not in selected and status not in ('done','cancelled')")
    db.execute("update jobs set status='pending' where id in selected and status in ('running','cancelled')")
    db.commit()
    rows=db.execute('select id,status,attempts,note from jobs where id in selected').fetchall()
    state={r[0]:r[1] for r in rows};attempts={r[0]:r[2] for r in rows};notes={r[0]:r[3] for r in rows}
    for t in tasks:
        if state[t['id']] != 'done' and (BASE/'receipts'/(t['id']+'.json')).exists():
            state[t['id']]='done'
            db.execute("update jobs set status='done' where id=?", (t['id'],))
    db.commit()
    start=time.time();active={};jobs={};last=0
    hand=BASE/'handover.json'
    adopted={};oldpids=[]
    if hand.exists():
        handover=json.loads(hand.read_text());oldpids=handover['pids']
        start=handover.get('started',start)
        for k in handover['tasks']:
            if k in all_ids and state[k]!='done':
                state[k]='running';adopted[k]=True;active[k]=handover.get('active',{}).get(k,time.time())
                db.execute("update jobs set status='running' where id=?", (k,))
        db.commit()
        hand.unlink()
    limits = POOL_LIMITS.copy()
    remaining_gpu = [t for t in tasks if t['resource'] in ('gpu', 'train')
                     and state[t['id']] not in ('done', 'failed', 'blocked')]
    if (remaining_gpu and CUDA_HOST_CPUS < os.sched_getaffinity(0)
            and all(t.get('inference_device') == 'cuda' for t in remaining_gpu)):
        limits['gpu'] = 3
    ctx=mp.get_context('spawn')
    pools={g:cf.ProcessPoolExecutor(max_workers=n,mp_context=ctx,initializer=init,initargs=(g!='gpu',)) for g,n in limits.items()}
    def track(k,s):
        state[k]=s;db.execute('update jobs set status=?, attempts=?,note=?,updated=? where id=?',(s,attempts[k],notes[k],time.time(),k))
    while True:
        old_live=any(proc_alive(pid) for pid in oldpids)
        for k in list(adopted):
            if (BASE/'receipts'/(k+'.json')).exists() or not old_live:
                track(k,'done' if (BASE/'receipts'/(k+'.json')).exists() else 'pending')
                del adopted[k];active.pop(k,None)
        for fut,(t,group) in list(jobs.items()):
            if not fut.done():continue
            k=t['id'];del jobs[fut];active.pop(k,None)
            try:ok,reason,sec=fut.result()
            except Exception as e:ok,reason=False,repr(e)
            notes[k]=reason
            track(k,'done' if ok else ('pending' if attempts[k]<2 else 'failed'))
        db.commit()
        import shutil
        used={g:sum(1 for t,p in jobs.values() if p==g) for g in pools}
        meminfo=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        working=int(Path('/sys/fs/cgroup/memory.current').read_text())-int(meminfo['inactive_file'])
        training_ready=any(t['resource']=='train' and state[t['id']]=='pending' and all(state[q]=='done' for q in t['deps']) for t in tasks)
        for job in tasks:
            if job['id'] in adopted:used['gpu' if job['resource']=='train' else job['resource']]+=1
        gpu_ready = sum(t['resource']=='gpu' and state[t['id']]=='pending'
                        and all(state[d]=='done' for d in t['deps']) for t in tasks)
        gpu_slots = min(limits['gpu'], used['gpu'] + gpu_ready)
        cpu_limit = min(limits['cpu'], 12 - min(gpu_slots, 2))
        for t in tasks:
            k=t['id']
            if state[k]!='pending':continue
            source=panel_sources.get(k)
            if source and state[source] in ('pending','running'):
                continue
            ds=[state[x] for x in t['deps']]
            if any(s in ('failed','blocked') for s in ds):
                notes[k]=next(x for x,s in zip(t['deps'],ds) if s in ('failed','blocked'));track(k,'blocked');continue
            if any(s!='done' for s in ds):continue
            cat=t['resource'];g='gpu' if cat=='train' else cat
            if used[g]>=(cpu_limit if g=='cpu' else limits[g]):continue
            if g=='gpu' and cat!='train' and training_ready:continue
            if g=='gpu' and ((cat=='train' and used[g]) or any(j['resource']=='train' for j,p in jobs.values())):continue
            if g!='light' and (working>21*2**30 or shutil.disk_usage(BASE).free<3*2**30):continue
            attempts[k]+=1;notes[k]='';track(k,'running');active[k]=time.time()
            jobs[pools[g].submit(run,t)]=(t,g);used[g]+=1
        db.commit()
        end=all(s in ('done','failed','blocked') for s in state.values())
        if time.time()-last>=5 or end:
            publish(tasks,state,active,notes,attempts,start);last=time.time()
        if end:break
        time.sleep(1)
    for p in pools.values():p.shutdown()
if __name__=='__main__':main()
