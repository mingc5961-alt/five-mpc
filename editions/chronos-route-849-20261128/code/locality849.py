"""Two NUMA-local byte-identical checkpoint mappings, using native scalar inference."""
from base849 import *
import shutil,time,torch
from filelock import FileLock
def source():return Path(get(OUT/'manifest.json')['paths']['Chronos/best.pt'])
def replica(node):
    d=OUT/'cache/native_numa';d.mkdir(parents=True,exist_ok=True);p=d/f'node{node}.pt'
    with FileLock(str(p)+'.lock'):
        if not p.exists():
            tmp=p.with_suffix('.part');shutil.copyfile(source(),tmp);os.replace(tmp,p)
        assert sha_file(p)==get(OUT/'manifest.json')['files'][str(source())]
    return p
def install(node):
    native=pf.chrono();state=torch.load(replica(node),weights_only=True,map_location='cpu',mmap=True)
    ret=native.model.load_state_dict(state,strict=True,assign=True)
    assert not ret.missing_keys and not ret.unexpected_keys
    native.model.eval()
def test(node):
    from grudata import schedule
    cores=[0,48];os.sched_setaffinity(0,{cores[node]})
    w,_=load_checked('confirmation',2000);m=Model(w,'Chronos');hs=[[] for _ in range(16)]
    for u,t in schedule(w,43000):
        hs[u].append(w.observe(u,t))
        if t>55:break
    t=time.monotonic();base=[m.base.update(u,h) for u,h in enumerate(hs)];before=time.monotonic()-t
    install(node);t=time.monotonic();ys=[m.base.update(u,h) for u,h in enumerate(hs)];after=time.monotonic()-t
    exact=sum(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(base,ys))
    r=dict(node=node,same_center_and_raw_width=exact,histories=16,original_s=before,mapped_s=after,checkpoint=sha_file(replica(node)))
    put(OUT/f'audits/numa_{node}.json',r);assert exact==16
if __name__=='__main__':test(int(sys.argv[1]))
