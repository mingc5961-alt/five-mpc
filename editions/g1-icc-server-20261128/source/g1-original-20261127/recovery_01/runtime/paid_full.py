"""Paid cold-start study. Immutable project components are loaded from --source."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
sys.dont_write_bytecode=True
warnings.filterwarnings("ignore",message="X does not have valid feature names")
SOURCE=Path(os.getenv("PAID_SOURCE","D:/低空论文/非规则航迹门控试验_20261124/仅改门控/自适应融合/四组闭环"))
OUT=Path(os.getenv("PAID_OUTPUT",str(Path(__file__).resolve().parent/"paid_full")))
sys.path.insert(0,str(SOURCE))
from simulate import model, State, raw, forecast, efficient_frontier
from common import BASE,SNAP,P,GRID,network,GROUPS,World,load_config,mixture,angle,difference
from pilot import Chronos
from routeclock.scheduler import TokenBucket,choose_actions
from routeclock.radio import Radio
from runtime.fast_ledger import DeferredLedger
import numpy as np

@lru_cache
def bundle():
    return pickle.loads((SOURCE/"tubes/G1.pkl").read_bytes())
@lru_cache
def chrono():
    c=Chronos(BASE/"training/Chronos-2/7/best.pt",SNAP)
    c.torch.set_num_threads(1)
    return c
def receipt(split,k,arm,m):
    return OUT/f"{split}_{k}_{arm}_{m:.2f}.json"

def run(job):
    split,k,arm,m=job
    dst=receipt(*job)
    if dst.exists():return
    then=time.monotonic()
    cfg=load_config()
    rho=.05/3
    cost=.0031
    cfg["simulation"]["sensing_budget"]=rho
    w=World(cfg,"paid_gru_horizon_v1/"+split,k,groups=[GROUPS[j] for j in mixture("irregular",k)])
    signature=hashlib.sha256(w.positions.tobytes()+np.asarray(w.shadows).tobytes()).hexdigest()
    radio=Radio(cfg)
    radio.threshold-=5
    ledger=DeferredLedger(cfg,w,radio,0.,180)
    bucket=TokenBucket(cost,rho)
    histories=[[] for _ in range(w.n)]
    slots=[SimpleNamespace(origin=0.,action=None,center=None,front=[]) for _ in histories]
    if arm=="C":
        cnet=chrono()
        factors=pickle.loads((BASE/"tubes/Chronos-2/7/bundle.pkl").read_bytes())["factors"]["native"]
    else:
        shared=model()
        if arm=="noGRU":
            import copy
            shared=copy.copy(shared)
            shared.residual=lambda history:np.zeros((20,3))
        tracks=[State(cfg,shared,"G") for _ in slots]
        factors=bundle()["final"]
    now=0.
    events=[]
    while now<180.:
        unused=next((u for u,h in enumerate(histories) if not h),None)
        due=[p.origin+(p.action.interval-cost if p.action else 0.) for p in slots]
        uid=unused if unused is not None else int(np.argmin(due))
        start=min(180.,max(now,now if unused is not None else due[uid],bucket.ready(now)))
        ledger.advance(now,start,slots,False)
        end=start+cost
        if end>180:
            ledger.advance(start,180.,slots,False)
            break
        bucket.consume(start)
        ledger.advance(start,end,slots,True)
        hist=histories[uid]
        obs=w.observe(uid,end)
        hist.append(obs)
        if arm=="C":
            center,width=cnet.batch([hist])[0]
            weights=[]
        else:
            tr=tracks[uid]
            pk=tr.update(obs,shared)["G"]
            center,features=forecast(pk,"G",1)
            short,route=pk[:2]
            weight=features[::2,5].copy()
            if arm!="G1":
                support=np.clip(features[::2,6],0,1)
                if arm=="flat":
                    weight=np.full_like(weight,weight[[2,4,10,20]].mean())
                    support=np.full_like(support,support[[2,4,10,20]].mean())
                weight=weight*support
                if arm=="noRoute":weight[:]=0
                center=short+weight[:,None]*(route-short)
                features[:,3]=np.abs(difference(angle(center),angle(obs.position))).ravel()
                features[:,5]=np.repeat(weight,2)
            width=raw(bundle()["tree"],features)
            weights=weight[[2,4,10,20]].tolist()
        p=slots[uid]
        p.center,p.origin=center,end
        p.front=efficient_frontier(radio,center,width,m*factors)
        fronts=[[] if i!=uid and (st.action is None or st.origin+st.action.interval<=end) else
                [a for a in st.front if st.origin+a.interval>end] for i,st in enumerate(slots)]
        for st,a in zip(slots,choose_actions(fronts,cost,rho),strict=True):
            st.action=a
        events.append([uid,end,len(hist),weights])
        now=end
    res=ledger.result(cost,rho)
    assert [e[0] for e in events[:16]]==list(range(16))
    assert abs(res["ledger_s"][0]-len(events)*cost)<1e-8 and res["ledger_s"][0]<=3.00001
    assert abs(sum(res["ledger_s"])-180)<1e-6 and abs(sum(res["bits"])/180e6-res["G_mbps"])<1e-7
    assert res["probes_per_user"]==9000
    tmp=dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(split=split,k=k,arm=arm,m=m,world=signature,result=res,events=events)))
    tmp.replace(dst)
    print(*job,round(res["G_mbps"],2),round(100*res["R"],3),round(time.monotonic()-then),flush=True)

def campaign(workers):
    OUT.mkdir(exist_ok=True,parents=True)
    arms=["F","C","G1","noGRU","noRoute","flat"]
    with ProcessPoolExecutor(workers) as pool:
        jobs=[("dev",k,a,m) for k in range(2) for m in (.75,1.) for a in arms]
        list(pool.map(run,jobs))
        scales={}
        for a in arms:
            candidates=[]
            for m in (.75,1.):
                rows=[json.loads(receipt("dev",k,a,m).read_text())["result"] for k in range(2)]
                g=np.mean([r["G_mbps"] for r in rows])
                r=np.mean([r["R"] for r in rows])
                candidates.append((r>.07,max(0,r-.07),-g,m))
            scales[a]=min(candidates)[-1]
        (OUT/"frozen.json").write_text(json.dumps(scales))
        list(pool.map(run,[("test",k,a,scales[a]) for k in range(8) for a in arms]))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args()
    campaign(a.workers)

