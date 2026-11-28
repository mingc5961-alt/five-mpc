"""Independent CPU work on exact paid prefixes. Worker count respects cgroup and memory caps."""
from chronos_recipients import *
from prefix849 import extend
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from policy_env import plan_events
from grudata import schedule
def capacity():
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    cpu=min(len(os.sched_getaffinity(0)),int(quota)//int(period) if quota!='max' else os.cpu_count())
    mem=int(Path('/sys/fs/cgroup/memory.max').read_text());used=int(Path('/sys/fs/cgroup/memory.current').read_text())
    # Shared model pages inherited on fork. Reserve 6 GiB and one core for the server.
    return max(1,min(cpu-1,int((mem-used-6*2**30)/(1.05*2**30))))
def busy(key):
    d=OUT/'workers';d.mkdir(exist_ok=True);p=d/str(os.getpid());atom(p,key.encode());return p
def warm(uid_stream):
    (u,seq),k=uid_stream;mark=busy('开发 1400 固定选人臂的 Chronos 预测缓存')
    try:
        w,wm=world('development',k);cache=Cache(w,wm,'development');mem=Memory()
        for a in seq:
            o=w.observe(u,a/1e6+COST);mem.on_paid_fix(o);cache(u,mem)
        cache.db.close()
        return (u,len(seq),cache.hits,cache.misses)
    finally:mark.unlink(missing_ok=True)
def warm_known():
    w,wm=world('development',1400);net=Model(w,'Chronos');streams=[]
    streams.append([[i%16,*event] for i,event in enumerate(cal)])
    for seed in (17,29,41):
        es,_=plan_events(w,grid,'fixed',policy('fixed',seed)[0]);streams.append(es)
    jobs=[((u,[a for uid,a,b in es if uid==u]),1400) for es in streams for u in range(16)]
    with ProcessPoolExecutor(min(36,capacity()),mp_context=mp.get_context('fork')) as pool:
        for v in pool.map(warm,sorted(jobs,key=lambda r:-len(r[0][1]))):print(v,flush=True)
def shard(job):
    split,k,parity=job
    from prefix849 import __file__ as source
    tag=busy(f'{split}/{k} E1 唯一付费前缀分片 {parity}')
    try:
        w,wm=world(split,k);model=Model(w,'Chronos');dest=OUT/f'E1/prefix_index/{split}/{k}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'receipt.json').exists():return
        spec=sha(compact([split,w.namespace,wm['physical_sha256'],modelid(),sha_file(Path(source))]))
        keys=[EMPTY]*16
        for i,(u,t) in enumerate(schedule(w,41000+k)):
            if u%2!=parity:continue
            o=w.observe(u,t);keys[u]=extend(keys[u],o);key=(spec,u,len(model.h[u])+1,o.t,keys[u]);p=dest/f'{i}.part'
            if p.exists():
                stored,c,width=pickle.loads(p.read_bytes());assert key==stored
                model.h[u].append(o);continue
            c,width=model.predict(u,o);atom(p,pickle.dumps((key,c,width),protocol=5))
    finally:tag.unlink(missing_ok=True)
if __name__=='__main__':warm_known()
