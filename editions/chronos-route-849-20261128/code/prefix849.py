"""A single frozen native inference per unique issued E1 prefix, with atomic checkpoints."""
from base849 import *
import pickle,time
from grudata import schedule
from functools import lru_cache
def obsbytes(o):
    return b''.join(np.asarray(v,dtype='<f8').tobytes() for v in ([o.t],o.position,o.covariance))
def extend(previous,o):return sha(bytes.fromhex(previous)+obsbytes(o))
EMPTY=sha(b'')
@lru_cache
def modelid():
    m=get(OUT/'manifest.json')
    return sha(compact([m['files'][m['paths'][p]] for p in ('Chronos/best.pt','Chronos/tube.pkl')]+[m['grid'],m['platform'],m['torch']]))
def collect(split,k):
    w,wm=world(split,k); model=Model(w,'Chronos');dest=OUT/'E1/prefix_index'/split/str(k);dest.mkdir(parents=True,exist_ok=True)
    entries=list(schedule(w,41000+k));hist=[EMPTY]*16;rows=[];seen=set()
    spec=sha(compact([split,w.namespace,wm['physical_sha256'],modelid(),sha_file(Path(__file__))]))
    rec=dest/'receipt.json';data=dest/'prefixes.pkl'
    if rec.exists():
        r=get(rec);assert r['spec']==spec and sha_file(data)==r['sha256'];return r
    start=time.monotonic()
    for i,(uid,t) in enumerate(entries):
        u=int(uid);o=w.observe(u,t);hist[u]=extend(hist[u],o);n=len(model.h[u])+1
        key=(spec,u,n,o.t,hist[u])
        if key in seen:raise ValueError('duplicate issuance')
        seen.add(key);file=dest/f'{i}.part'
        if file.exists():
            prev,c,width=pickle.loads(file.read_bytes())
            assert key==prev;model.h[u].append(o)
        else:
            c,width=model.predict(u,o);atom(file,pickle.dumps((key,c,width),protocol=5))
        assert c.shape==(21,3) and width.shape==(21,2) and np.isfinite(c).all() and np.isfinite(width).all()
        rows.append((u,o,n,hist[u],arrhash(c),c,width))
    atom(data,pickle.dumps(rows,protocol=5))
    r=dict(spec=spec,sha256=sha_file(data),split=split,world=k,physics=wm['physical_sha256'],namespace=w.namespace,prefixes=len(rows),dtype=str(rows[0][-2].dtype),weights=modelid(),grid=pf.GRID.tolist(),wall_seconds=time.monotonic()-start)
    put(rec,r)
    for f in dest.glob('*.part'):f.unlink()
    return r
if __name__=='__main__':print(collect(sys.argv[1],int(sys.argv[2])),flush=True)
