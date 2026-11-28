"""A paid observation can only update forecasts issued for that aircraft."""
from grudata import *
from grutrain import Cold
from paid_full import State, model, forecast,raw,bundle,chrono,BASE,efficient_frontier,TokenBucket,Radio,choose_actions,DeferredLedger,angle,difference
from types import SimpleNamespace
from functools import lru_cache
import hashlib,json,time
@lru_cache
def cold(mode):return Cold(mode)
class Predictor:
    def __init__(self,cfg,arm):
        self.arm=arm
        if arm=="C":
            self.net=chrono(); self.net.torch.set_num_threads(1)
            self.factors=pickle.loads((BASE/"tubes/Chronos-2/7/bundle.pkl").read_bytes())["factors"]["native"]
        else:
            self.net=model() if arm=="G1" else cold(arm)
            if arm=="NoGRU":
                from copy import copy
                self.net=copy(self.net)
                self.net.residual=lambda history: np.zeros((20,3))
            self.tr=[State(cfg,self.net,"G") for _ in range(16)]
            self.factors=bundle()["final"]
    def update(self,u,h):
        if self.arm=="C":return self.net.batch([h])[0]
        self.net.n=len(h)
        pk=self.tr[u].update(h[-1],self.net)["G"]
        c,f=forecast(pk,"G",1)
        if self.arm in ("Static","NoRoute"):
            # Same prior and final profile as G1, remove only its realized-error feedback.
            shift=np.interp(GRID,[0,1,2,5,10],[0,0,0,.05,.10])
            weight=np.zeros_like(GRID) if self.arm=="NoRoute" else np.clip(self.net.prior+shift,0,1)
            c=pk[0]+weight[:,None]*(pk[1]-pk[0])
            f[:,3]=abs(difference(angle(c),angle(h[-1].position))).ravel()
            f[:,5]=np.repeat(weight,2)
        return c,raw(bundle()["tree"],f)
def path(s,k,a,m):return D/f"run_{s}_{k}_{a}_{m:.2f}.json"
def run(job):
    s,k,a,m=job
    dest=path(*job)
    if dest.exists():return
    w=world(s,k); cfg=w.cfg;net=Predictor(cfg,a)
    rho=.05/3;cost=.0031
    radio=Radio(cfg);radio.threshold-=5
    ledger=DeferredLedger(cfg,w,radio,0.,180)
    bucket=TokenBucket(cost,rho)
    histories=[[] for _ in range(16)]
    slots=[SimpleNamespace(origin=0.,action=None,center=None,front=[]) for _ in histories]
    now=0;events=[]
    while now<180:
        first=next((u for u,h in enumerate(histories) if not h),None)
        due=[v.origin+(v.action.interval-cost if v.action else 0.) for v in slots]
        u=first if first is not None else int(np.argmin(due))
        start=min(180.,max(now,now if first is not None else due[u],bucket.ready(now)))
        ledger.advance(now,start,slots,False)
        end=start+cost
        if end>180:
            ledger.advance(start,180.,slots,False)
            break
        bucket.consume(start); ledger.advance(start,end,slots,True)
        h=histories[u];h.append(w.observe(u,end))
        c,width=net.update(u,h)
        p=slots[u];p.center=c;p.origin=end
        p.front=efficient_frontier(radio,c,width,m*net.factors)
        fronts=[[] if v is not p and (v.action is None or v.origin+v.action.interval<=end) else [b for b in v.front if v.origin+b.interval>end] for v in slots]
        for v,b in zip(slots,choose_actions(fronts,cost,rho),strict=True):v.action=b
        events.append([u,end]);now=end
    r=ledger.result(cost,rho)
    assert abs(sum(r["bits"])/180e6-r["G_mbps"])<1e-6 and r["probes_per_user"]==9000
    assert abs(r["ledger_s"][0]-len(events)*cost)<1e-7 and sum(r["ledger_s"][:1])<=3.00001
    assert [e[0] for e in events[:16]]==list(range(16))
    fp=hashlib.sha256(w.positions.tobytes()+np.asarray(w.noise).tobytes()+np.asarray(w.shadows).tobytes()).hexdigest()
    tmp=dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(world=fp,split=s,k=k,arm=a,m=m,result=r,events=events)));tmp.replace(dest)
    print(*job,round(r["G_mbps"],1),round(100*r["R"],3),flush=True)
