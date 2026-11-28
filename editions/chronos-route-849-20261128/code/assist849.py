"""Use idle cores on pending, deduplicated counterfactual branches."""
from labels849 import *
from parallel849 import capacity
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing
@lru_cache(maxsize=2)
def physical(split,k):
    w,wm=world(split,k);return w,Cache(w,wm,split)
@lru_cache(maxsize=1)
def restored(path):
    root=Path(path);meta=get(root/'snapshot.json');w,cache=physical(meta['split'],meta['world']);p=root/'snapshot.pkl'
    assert sha_file(p)==meta['snapshot_sha']
    c=Engine(w,cache,.75,False,'EDF-native');c.__dict__.update(pickle.loads(p.read_bytes()));c.ledger.rr=meta['rr']
    assert sha(compact([m.key for m in c.mem]))==meta['prefix']
    return c,meta,root
def helpone(job):
    path,u=job;c,meta,root=restored(path)
    r=one_candidate(c,meta,root,u,blocking=False)
    return r is not None
def host():
    while not any((OUT/'E2/labels').glob('*/*/*/snapshot.json')):
        if (OUT/'E2/labels/VALUE-time.txt').is_file():return
        time.sleep(5)
    w,_=world('development',1400);Model(w,'Chronos')
    with ProcessPoolExecutor(capacity(),mp_context=multiprocessing.get_context('fork')) as p:
        active={}
        while True:
            if (OUT/'E2/labels/VALUE-time.txt').is_file():break
            cores=capacity();owners=0
            for f in (OUT/'workers').glob('*'):
                try:
                    if Path('/proc',f.name).exists():owners+=1
                except OSError:pass
            room=max(0,cores-max(1,owners)-len(active))
            if room:
                in_use=set(active.values());todo=[]
                for meta in sorted((OUT/'E2/labels').glob('*/*/*/snapshot.json'),key=lambda p:p.stat().st_mtime,reverse=True):
                    root=meta.parent
                    if (root/'state.json').exists():continue
                    for u in reversed(range(16)):
                        j=(str(root),u)
                        if j not in in_use and not (root/f'{u}.json').exists():todo.append(j)
                for j in todo[:room]:active[p.submit(helpone,j)]=j
            if active:
                done,_=wait(active,timeout=3,return_when=FIRST_COMPLETED)
                for f in done:f.result();active.pop(f)
            else:time.sleep(3)
if __name__=='__main__':host()
