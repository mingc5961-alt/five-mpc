from factorial_heads import *
import grurun
from itertools import combinations
from functools import lru_cache
from unittest.mock import patch

BASEP=grurun.Predictor
@lru_cache(None)
def trained():
    return pickle.loads((ROOT/"heads.pkl").read_bytes())

class Adapter:
    def __init__(self,cfg,arm):
        self.arm=arm
        if arm in ("G1","C"):
            self.base=BASEP(cfg,arm)
            self.factors=self.base.factors
        else:
            self.track=Track(cfg)
            self.beta,self.heads=trained()
            self.factors=np.array(json.loads((ROOT/"factors.json").read_text())[arm])
    def update(self,u,h):
        if self.arm in ("G1","C"):return self.base.update(u,h)
        c,f,e,z=self.track.step(u,h[-1])
        width=raw(bundle()["tree"],f)
        mode=int(self.arm.startswith("R"))
        if self.arm.endswith("1"):
            x=inputs(f[None],e[None],z[None],width[None])
            width=cond_width(self.heads[mode],x,width)
        return (adjusted(c,e,self.beta) if mode else c),width

def receipt(split,k,a,m):
    if split=="risk" and a in ("G1","C"):
        return ROOT.parent/"g1mix7"/f"run_test_{k}_{a}_{m:.2f}.json"
    return ROOT/f"run_{split}_{k}_{a}_{m:.2f}.json"

def simulate(job):
    dst=receipt(*job)
    if dst.exists():return
    lock=dst.with_suffix(".lock")
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        while lock.exists() and not dst.exists():time.sleep(2)
        if not dst.exists():return simulate(job)
        return
    try:
        run_locked(job)
    finally:
        lock.unlink(missing_ok=True)

def run_locked(job):
    with patch.object(grurun,"D",ROOT),patch.object(grurun,"world",fresh_world),patch.object(grurun,"Predictor",Adapter),patch.object(grurun,"path",receipt):
        grurun.run(job)

def policy(a,grid):
    pts={}
    for m in grid:
        runs=[json.loads(receipt("risk",k,a,m).read_text())["result"] for k in range(8)]
        pts[m]=np.mean([[r["G_mbps"],r["R"]*100] for r in runs],axis=0)
    options=[]
    for (m,(g,r)),(n,(v,s)) in combinations(pts.items(),2):
        if abs(r-s)<1e-10:continue
        p=(7-s)/(r-s)
        if 0<=p<=1:options.append((p*g+(1-p)*v,m,n,p))
    if not options:return None
    _,m,n,p=max(options)
    return {str(m):p,str(n):1-p}

if __name__=="__main__":
    waitfile(ROOT/"factors.json")
    for k in range(8):waitfile(ROOT/f"eval_confirm{k}.npz")
    arms=(*ARMS,"G1","C")
    grids={a:[.40,.85] for a in ARMS}
    grids.update(G1=[.45,.75],C=[.75,1.])
    with ProcessPoolExecutor(4) as pool:
        jobs=[("risk",k,a,m) for k in range(8) for a in ARMS for m in grids[a]]
        list(pool.map(simulate,jobs))
        plans={}
        for a in arms:
            for m in (.29,1.,.20,1.25,.12):
                if policy(a,grids[a]) is not None:break
                if m in grids[a]:continue
                list(pool.map(simulate,[("risk",k,a,m) for k in range(8)]));grids[a].append(m)
            plans[a]=policy(a,grids[a])
            if plans[a] is None:raise ValueError(("no 7% bracket",a))
        (ROOT/"freeze.json").write_text(json.dumps(plans))
        print("frozen",plans,flush=True)
        jobs=[("confirm",k,a,float(m)) for k in range(8) for a in arms for m in plans[a]]
        jobs.extend(("confirm",k,"G1",1.) for k in range(8))
        list(pool.map(simulate,list(dict.fromkeys(jobs))))
    for k in range(8):
        w=fresh_world("confirm",k)
        events=json.loads(receipt("confirm",k,"G1",.75).read_text())["events"]
        save_schedule(w,k,Track(w.cfg),events,ROOT/f"data_shadow{k}.npz")
        evaluate(("shadow",k))
