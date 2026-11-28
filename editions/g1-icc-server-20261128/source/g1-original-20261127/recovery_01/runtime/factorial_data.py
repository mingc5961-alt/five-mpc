from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pickle
from grudata import World, GROUPS, load_config, GRID, np, schedule, truth
from paid_full import State, model, forecast, bundle, raw, angle, difference, network

ROOT=Path(__file__).resolve().parent/"factorial_run"
ROOT.mkdir(exist_ok=True)
HS=np.asarray(network.QUERY_H[1:])
ARMS=("O0","O1","R0","R1")
DENSE=np.arange(501)*.02
HI=np.searchsorted(GRID,DENSE-1e-10).clip(0,20)
LO=np.where(np.isclose(GRID[HI],DENSE),HI,(HI-1).clip(0))
BETA_ROOT=ROOT.parent/"g1_transfer_probe"

def get_beta():
    arr=np.concatenate([np.load(BETA_ROOT/f"train{k}.npy") for k in range(12)])
    slope=np.clip((arr[:,0]*arr[:,1]).sum((0,2))/(arr[:,1]**2).sum((0,2)),-1,1)
    return np.interp(GRID,[0,1,2,5,10],np.r_[0,slope])

def fresh_world(split,k):
    if split=="risk":
        from grudata import world
        return world("test",k)
    path=ROOT/f"{split}{k}.world"
    if path.exists():return pickle.loads(path.read_bytes())
    cfg=load_config();cfg["simulation"].update(warmup_s=0,service_s=180,sensing_budget=.05/3)
    rng=np.random.default_rng(410827+k)
    groups=np.repeat(np.arange(6),[3,1,1,1,3,7])[rng.permutation(16)]
    w=World(cfg,"signed_g1_factorial_v1/"+split,k,groups=[GROUPS[i] for i in groups])
    tmp=path.with_suffix(".tmp");tmp.write_bytes(pickle.dumps(w,protocol=5));tmp.replace(path)
    return w

def adjusted(c,e,beta):
    if not np.any(e):return c.copy()
    theta=angle(c)+beta[:,None]*e
    az,el=theta.T
    return np.linalg.norm(c,axis=1)[:,None]*np.column_stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])

class Track:
    def __init__(self,cfg):
        self.net=model()
        self.state=[State(cfg,self.net,"G") for _ in range(16)]
        self.last=[None]*16
    def step(self,u,o):
        c,f=forecast(self.state[u].update(o,self.net)["G"],"G",1)
        old=self.last[u];self.last[u]=(o.t,c)
        valid=old is not None and o.t-old[0]<=10
        e=np.zeros(2)
        if valid:
            xyz=[np.interp(o.t-old[0],GRID,a) for a in old[1].T]
            e=difference(angle(o.position),angle(np.asarray(xyz)))
        return c,f,e,np.array([np.log1p(self.state[u].count),float(valid)])

def extract(job):
    split,k=job
    dst=ROOT/f"data_{split}{k}.npz"
    if dst.exists():return
    w=fresh_world(split,k);track=Track(w.cfg)
    if split=="shadow":
        raise ValueError("use extract_shadow")
    save_schedule(w,k,track,schedule(w,41000+k),dst)

def save_schedule(w,k,track,events,dst):
    cs=[];fs=[];es=[];zs=[];us=[];ts=[]
    for u,t in events:
        c,f,e,z=track.step(u,w.observe(u,t))
        cs.append(c);fs.append(f);es.append(e);zs.append(z);us.append(u);ts.append(t)
    np.savez_compressed(dst,c=cs,f=np.asarray(fs,dtype=np.float32),e=es,z=zs,u=us,t=ts)

if __name__=="__main__":
    with ProcessPoolExecutor(3) as p:
        list(p.map(extract,[(s,k) for s in ("fit","tube","confirm") for k in range(8)]))
