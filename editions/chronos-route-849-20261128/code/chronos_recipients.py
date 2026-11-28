"""Fixed-calendar causal recipient selection with unchanged native post-purchase service."""
from chronos_route_heads import *
from prefix849 import modelid
from types import SimpleNamespace
from functools import cached_property
import copy,gzip,sqlite3,time
from routeclock.geometry import angular_jacobian
from routeclock.filters import PositionIMM
from learned_matrix import policy
from aoi_policy import PaidState
import torch
COST=.0031;RHO=3/180
MATRIX=[('MAXAGE',0),*(('AoI-DQN-fixed',s) for s in (17,29,41)),('VAR',0),('EDF-native',0),('EDF-time',0),('EDF-route',0),('VALUE-time',0),('VALUE-route',0)]
TICKS=list(grid.fixed_indices())
def tie(scores,origins,native=False):
    best=np.flatnonzero(scores==np.max(scores))
    return int(best[0] if native else min(best,key=lambda u:(origins[u],u))),len(best)
def variance(track,age):
    if track.t is None:return 0.
    pos,cov=track.forecast([round(float(age),6)]);p=cov[0];j=angular_jacobian(pos[0])
    # Public sensor model at the predicted location, with no hypothetical observation sampled.
    from routeclock.world import position_covariance
    from routeclock.geometry import angles
    m=track.std
    noise=position_covariance(np.r_[np.linalg.norm(pos[0]),angles(pos[0]),0.],m)
    return float(np.trace(j@p@np.linalg.solve(p+noise,p)@j.T))
@lru_cache
def value_head(which):
    from lightgbm import Booster
    return Booster(model_file=str(OUT/f'E2/labels/VALUE-{which}.txt'))
class Cache:
    def __init__(self,w,wm,split):
        self.model=Model(w,'Chronos');self.hits=self.misses=0
        self.spec=sha(compact([split,w.namespace,wm['physical_sha256'],modelid(),str(pf.GRID),'native-single-float32',sha_file(Path(__file__).with_name('chronos_route_heads.py'))]))
        folder=OUT/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(folder/(self.spec+'.sqlite'),timeout=120,isolation_level=None)
        self.db.execute('pragma busy_timeout=120000')
        self.db.execute('create table if not exists forecast(k text primary key, sig text not null, payload blob not null)')
    def __call__(self,u,mem):
        key=sha(compact([self.spec,u,len(mem.h),mem.h[-1].t,mem.key]))
        row=self.db.execute('select sig,payload from forecast where k=?',(key,)).fetchone()
        if row:
            checksum,data=row
            assert sha(data)==checksum
            self.hits+=1;return pickle.loads(data)
        self.misses+=1
        result=self.model.base.update(u,mem.h)
        data=pickle.dumps(result,protocol=5);sig=sha(data)
        self.db.execute('insert or ignore into forecast values (?,?,?)',(key,sig,data))
        assert self.db.execute('select sig from forecast where k=?',(key,)).fetchone()[0]==sig
        return result
class Engine:
    def __init__(self,w,cache,m,smoke,rule,seed=0):
        self.w=w;self.cache=cache;self.m=m;self.smoke=smoke;self.rule=rule
        self.radio=pf.Radio(w.cfg);self.radio.threshold-=5
        self.ledger=pf.DeferredLedger(w.cfg,w,self.radio,0.,180.);self.bucket=pf.TokenBucket(COST,RHO)
        self.slots=[SimpleNamespace(origin=0.,action=None,center=None,front=[]) for _ in range(16)]
        self.mem=[Memory() for _ in self.slots];self.risk=np.zeros((16,2));self.sup=np.zeros((16,3),int)
        self.filters=[PositionIMM() for _ in self.slots];s=w.cfg['simulation']['measurement_std']
        for f in self.filters:f.std=np.array([s['range_m'],np.deg2rad(s['azimuth_deg']),np.deg2rad(s['elevation_deg']),s['radial_velocity_m_s']])
        self.now=0.;self.events=[];self.audit=[];self.issued=[];self.i=0
        self.dqn=PaidState(grid,'fixed') if rule=='AoI-DQN-fixed' else None
        self.net=policy('fixed',seed)[0] if self.dqn else None
    def deadline(self):
        return np.asarray([s.origin+(s.action.interval-COST if s.action else 0.) for s in self.slots])
    def view(self,start):
        origins=np.asarray([s.origin for s in self.slots]);ages=start-origins;dead=self.deadline()
        return origins,ages,dead
    def values(self,start,which,view):
        origin,age,dead=view;kind=int(which=='route')
        rem=origin+self.risk[:,kind]-start
        dur=np.array([s.action.interval if s.action else 0. for s in self.slots])
        pair=np.array([np.prod(s.action.pair) if s.action else 0. for s in self.slots])
        n=np.array([len(mem.h) for mem in self.mem]);sup=self.sup[:,kind]
        c=np.column_stack([age,dead-start,rem,self.risk[:,kind],dur,pair,n,sup])
        return np.column_stack([c,np.tile([self.i/849,age.mean(),age.max(),rem.mean(),(rem<0).mean()],(16,1))])
    def score(self,rule,start,view):
        origin,age,dead=view
        if rule=='MAXAGE':return age
        if rule=='EDF-native':return -dead
        if rule=='VAR':return np.array([variance(f,a) for f,a in zip(self.filters,age)])
        if rule.startswith('EDF-'):return -(origin+self.risk[:,int(rule.endswith('route'))])
        if rule.startswith('VALUE-'):
            which=rule.split('-')[1];x=self.values(start,which,view)
            return x[:,0]-.4*x[:,2]+.01*x[:,7] if self.smoke else value_head(which).predict(x,num_threads=1)
        if rule=='AoI-DQN-fixed':
            while self.dqn.tick<TICKS[self.i]:
                _,_,legal=self.dqn.view();self.dqn.step(16,None,legal)
            users,public,legal=self.dqn.view()
            with torch.inference_mode():return self.net(torch.from_numpy(users)[None],torch.from_numpy(public)[None])[0,:16].numpy()
        raise ValueError(rule)
    def preservice(self,i):
        start=cal[i][0]/1e6
        self.ledger.advance(self.now,start,self.slots,False);self.now=start
    def buy(self,i,u,heads=True):
        start,end_us=cal[i];start=start/1e6;end=start+COST
        assert self.now==start and round(end*1e6)==end_us and start+1e-10>=self.bucket.ready(start)
        self.bucket.consume(start);self.ledger.advance(start,end,self.slots,True)
        obs=self.w.observe(u,end);mem=self.mem[u];mem.on_paid_fix(obs);c,width=self.cache(u,mem)
        if self.rule=='VAR':self.filters[u].observe(obs)
        if heads:
            q=mem.query();self.sup[u]=q[3:]
            effective={}
            for j,arm in enumerate(('TIME','ROUTE')):
                eff=inherited(arm,q)
                if eff not in effective:
                    sw,sf,_=issued(arm,mem,width,q,self.smoke);effective[eff]=admitted(self.radio,c,sw,sf,self.m)
                self.risk[u,j]=effective[eff]
        mem.issue(c)
        s=self.slots[u];s.origin=end;s.center=c;s.front=pf.efficient_frontier(self.radio,c,width,self.m*self.cache.model.factors)
        fronts=[[] if v is not s and (v.action is None or v.origin+v.action.interval<=end)
                else [a for a in v.front if v.origin+a.interval>end] for v in self.slots]
        for v,a in zip(self.slots,pf.choose_actions(fronts,COST,RHO),strict=True):v.action=a
        if self.dqn:
            _,_,legal=self.dqn.view();self.dqn.step(u,lambda *_:obs,legal)
        self.now=end;self.events.append([u,round(start*1e6),round(end*1e6)])
        self.issued.append((u,mem.key,arrhash(c),sha(obsfull(obs))))
        self.i=i+1
    def decision(self,i,detail=True):
        self.preservice(i);t=self.now;view=self.view(t);dqn=self.rule=='AoI-DQN-fixed'
        scores=self.score(self.rule,t,view)
        uid,ties=tie(scores,view[0],self.rule in ('EDF-native','AoI-DQN-fixed','MAXAGE'))
        if i<16:uid=i
        if detail:
            same={}
            if self.rule=='VALUE-route' and i>=16:
                for rule in ('VALUE-time','EDF-native','MAXAGE'):
                    ss=self.score(rule,t,view);same[rule]=tie(ss,view[0],rule!='VALUE-time')[0]
            self.audit.append(dict(i=i,tick=TICKS[i],uid=uid,pre_history_sha=[m.key for m in self.mem],scores=scores.tolist(),age=view[1].tolist(),deadline=view[2].tolist(),reference_deadlines=(view[0][:,None]+self.risk).tolist(),support=self.sup.tolist(),tie=ties,fallback=(self.sup[:,0]==0).tolist(),same_state=same))
        self.buy(i,uid)
    def result(self):
        self.ledger.advance(self.now,180.,self.slots,False)
        r=self.ledger.result(COST,RHO)
        assert [v[1:] for v in self.events]==cal and [e[0] for e in self.events[:16]]==list(range(16))
        assert len(r['ledger_s'])==4 and abs(sum(r['ledger_s'])-180)<1e-6 and abs(r['ledger_s'][0]-2.6319)<1e-7
        assert r['probes_per_user']==9000 and abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-6
        return r
def endpoint(job,cache_factory=Cache):
    phase,split,k,rule,seed,m=job
    stem=f'{k}_{rule}_s{seed}_m{m:.2f}';folder=OUT/'E2/receipts'/phase;dest=folder/(stem+'.json')
    lp=OUT/'E2/decision_logs'/phase/(stem+'.json.gz')
    w,wm=world(split,k);spec=sha(compact([phase,wm['physical_sha256'],service_id(),run_id(phase),rule,seed,m]))
    if dest.exists():
        r=get(dest);assert r['spec']==spec and r['log_sha']==sha_file(lp) and [e[1:] for e in r['events']]==cal
        return stem
    cache=cache_factory(w,wm,split);eng=Engine(w,cache,m,phase=='smoke',rule,seed)
    start=time.monotonic()
    activity=OUT/'active_closed'/phase/(stem+'.json')
    for i in range(849):
        eng.decision(i)
        if i%64==63 or i==848:
            put(activity,dict(pid=os.getpid(),world=k,arm=rule,completed_fixes=i+1,nominal=849,elapsed_s=time.monotonic()-start))
    r=eng.result()
    if phase in ('smoke','final-smoke') and rule in ('MAXAGE','EDF-native'):
        hist=[Memory() for _ in eng.mem]
        def pred(u,o):
            hist[u].on_paid_fix(o);return cache(u,hist[u])
        es,old,_=closed.simulate_controls(w,'oldest' if rule=='MAXAGE' else 'fixed_due',pred,cache.model.factors,pf,3.,m,849)
        assert [[u,round(end*1e6)] for u,end in es]==[[u,e] for u,_,e in eng.events] and old==r
    payload=dict(service_model_sha=service_id(),calendar_sha=P['scene']['calendar_sha256'],decisions=eng.audit,issued=eng.issued)
    atom(lp,gzip.compress(compact(payload),compresslevel=2,mtime=0))
    put(dest,dict(spec=spec,phase=phase,world=k,physics=wm['physical_sha256'],arm=rule,seed=seed,scale=m,result=r,events=eng.events,event_sha=sha(compact(eng.events)),log_sha=sha_file(lp),service_model_sha=service_id(),elapsed_s=time.monotonic()-start,cache_hits=cache.hits,cache_misses=cache.misses))
    activity.unlink(missing_ok=True)
    return stem
@lru_cache
def service_id():
    import inspect
    return sha(compact([modelid(),*[sha(inspect.getsource(v).encode()) for v in (Engine.buy,pf.efficient_frontier,pf.choose_actions,pf.DeferredLedger,pf.TokenBucket)]]))
def run_id(phase):
    if phase=='smoke':return 'initial-wiring-v1'
    if phase in ('final-smoke','confirmation'):return sha_file(OUT/'design/frozen.json')
    return sha(compact([sha_file(OUT/f'E1/calibration/{a}.pkl') for a in LEARN]+[sha_file(OUT/f'E2/labels/VALUE-{j}.txt') for j in ('time','route')]))
if __name__=='__main__':
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    w,wm=world('development',1400);cache=Cache(w,wm,'development');cache.db.close()
    with ProcessPoolExecutor(10,mp_context=mp.get_context('fork')) as pool:
        for name in pool.map(endpoint,[('smoke','development',1400,r,s,.75) for r,s in MATRIX]):print(name,flush=True)
