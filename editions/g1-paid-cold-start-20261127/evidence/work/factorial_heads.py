from factorial_data import *
from lightgbm import LGBMRegressor
import json,time

def inputs(f,e,z,width):
    # One row per horizon/angle. Residual quantiles are learned in units of the G1 scale.
    aux=np.repeat(np.concatenate((np.rad2deg(e),z),axis=1)[:,None,:],42,axis=1)
    return np.concatenate((f,aux,np.log(width.reshape(-1,42,1))),axis=2)

def centers(data,beta):
    return np.stack([adjusted(c,e,beta) for c,e in zip(data["c"],data["e"])])

def learn():
    beta=get_beta();bx=[];ys=[[],[]]
    for k in range(8):
        w=fresh_world("fit",k)
        with np.load(ROOT/f"data_fit{k}.npz") as a:
            c=a["c"];f=a["f"];width=raw(bundle()["tree"],f)
            bx.append(inputs(f,a["e"],a["z"],width))
            label=np.stack([angle(truth(w,u,t+GRID)) for u,t in zip(a["u"],a["t"])])
            for j,center in enumerate((c,centers(a,beta))):
                ys[j].append(difference(label,angle(center))/width)
    x=np.concatenate(bx).reshape(-1,bx[0].shape[-1])
    heads=[]
    for labels in ys:
        y=np.concatenate(labels).reshape(-1)
        pair=[]
        for sign in (-1,1):
            tree=LGBMRegressor(objective="quantile",alpha=.9,n_estimators=65,
                learning_rate=.055,num_leaves=12,min_child_samples=160,n_jobs=2,
                random_state=2701,verbosity=-1)
            tree.fit(x,sign*y);pair.append(tree.booster_)
        heads.append(pair)
    return beta,heads

def cond_width(head,x,width):
    xx=x.reshape(-1,x.shape[-1])
    ratio=np.maximum(head[0].predict(xx,num_threads=1),head[1].predict(xx,num_threads=1))
    return width*np.clip(ratio.reshape(width.shape),.15,8.)

def evaluate(job):
    split,k=job
    with (ROOT/"heads.pkl").open("rb") as s:beta,heads=pickle.load(s)
    w=fresh_world("confirm" if split=="shadow" else split,k)
    with np.load(ROOT/f"data_{split}{k}.npz") as d:
        f=d["f"];e=d["e"];z=d["z"];width=raw(bundle()["tree"],f)
        x=inputs(f,e,z,width)
        c=[d["c"],centers(d,beta)]
        wid=[width,cond_width(heads[0],x,width),width,cond_width(heads[1],x,width)]
        errs=[];scores=[];areas=[]
        for i,(u,t) in enumerate(zip(d["u"],d["t"])):
            target=angle(truth(w,u,t+DENSE));res=[]
            for cc in c:
                dense=np.column_stack([np.interp(DENSE,GRID,col) for col in cc[i].T])
                res.append(abs(difference(target,angle(dense))))
            errs.append(np.rad2deg(np.linalg.norm(np.array(res)[:,::25],axis=-1)))
            sc=[];area=[]
            for j,ww in enumerate(wid):
                bound=np.maximum(ww[i,HI],ww[i,LO])
                ratio=np.max(res[j//2]/bound,axis=1)
                sc.append(np.maximum.accumulate(ratio)[np.rint(HS*50).astype(int)])
                area.append(np.mean(4*np.prod(np.rad2deg(bound),axis=1)))
            scores.append(sc);areas.append(area)
        np.savez_compressed(ROOT/f"eval_{split}{k}.npz",err=errs,score=scores,area=areas,z=z,t=d["t"],u=d["u"])

def waitfile(p):
    while not p.exists():time.sleep(1.)

if __name__=="__main__":
    # The producer can still be writing the last archive; wait for file integrity.
    from zipfile import ZipFile,BadZipFile
    def ready(s):
        for k in range(8):
            f=ROOT/f"data_{s}{k}.npz";waitfile(f)
            while True:
                try:
                    with ZipFile(f) as z: assert z.testzip() is None
                    break
                except (BadZipFile,OSError):time.sleep(1)
    ready("fit")
    (ROOT/"g1_original_factors.json").write_text(json.dumps(bundle()["final"].tolist()))
    with (ROOT/"heads.pkl").open("wb") as f:pickle.dump(learn(),f)
    with ProcessPoolExecutor(3) as pool:
        ready("tube")
        list(pool.map(evaluate,[("tube",k) for k in range(8)]))
        s=np.concatenate([np.load(ROOT/f"eval_tube{k}.npz")["score"] for k in range(8)])
        q=np.maximum.accumulate(np.quantile(s,.9,axis=0,method="higher"),axis=-1)
        (ROOT/"factors.json").write_text(json.dumps(dict(zip(ARMS,q.tolist()))))
        print("factors",q,flush=True)
        ready("confirm")
        list(pool.map(evaluate,[("confirm",k) for k in range(8)]))
