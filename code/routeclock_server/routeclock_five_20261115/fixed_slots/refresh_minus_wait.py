"""Paired refresh/wait labels and fixed-clock comparison against frozen baselines."""
from prepare import ROOT, OUT, COST, N, PERIOD, QUANTUM, ROUNDS, SLOTS
from compare import features, load_pickle, publish, new_observation, slot_trace
from routeclock.predictors import Tracker
from routeclock.world import World
from routeclock.config import load_config
from routeclock.radio import Radio
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import pickle
import sys
import numpy as np
import pandas as pd

DEST = ROOT / "runs" / "fixed_slots_refresh_minus_wait_cpu"
def objects(world):
    return load_pickle(ROOT/"runs/five_pilot/results/bundle.pkl"), Radio(world.cfg)

def initial_states(trackers, bundle, radio, cfg):
    return [publish(tr.evidence.archive[-1], bundle, radio, cfg) for tr in trackers]

def order_and_offsets(rng):
    order = np.argsort(rng.permutation(N))
    return order, COST + (order + N*(ROUNDS-1)/2 + .5)*QUANTUM

def valid_count(state,t,idx,switch):
    if state.action is None:
        return 0
    mid = t + COST + (idx + N*np.arange(ROUNDS))*QUANTUM + (QUANTUM+switch)/2
    return int(sum(mid <= state.package.origin + 10 + 1e-9))

def label(world,bundle,radio,tracker,st,t,i,order,off,k,source):
    X=features(st,t,off)
    tr, fresh=new_observation(tracker,i,t,world,bundle,radio)
    wait=slot_trace(world,radio,i,st,t,order)
    refresh=slot_trace(world,radio,i,fresh,t,order)
    nvalid=valid_count(st,t,order,world.cfg["simulation"]["beam_switch_s"])
    assert nvalid==ROUNDS-wait[2]
    row=dict(source=source,mission=world.mission,uid=i,slot=k,valid=nvalid,
             wait=wait[0],refresh=refresh[0])
    row.update({f"x{j}":v for j,v in enumerate(X)})
    return row,tr,fresh

def train_part(key):
    tag,m=key
    rows=[]
    if tag=="replay":
        world,trackers=load_pickle(OUT/f"train_{m}.pkl")
        bundle,radio=objects(world)
        states=initial_states(trackers,bundle,radio,world.cfg)
        rng=np.random.default_rng(8100+m)
        order,offset=order_and_offsets(rng)
        for k in range(SLOTS):
            t=180+k*PERIOD
            ages=[t-st.package.origin for st in states]
            picked=int(rng.integers(N)) if rng.random()<.65 else int(np.argmax(ages))
            ids=sorted(set([picked,int(np.argmax(ages)),*rng.choice(N,2,replace=False)]))
            for i in ids:
                row,tr,st=label(world,bundle,radio,trackers[i],states[i],t,i,order[i],offset[i],k,tag)
                rows.append(row)
                if i==picked:
                    trackers[i],states[i]=tr,st
        df=pd.DataFrame(rows)
        prev=pd.read_csv(OUT/f"train_{m}_train.csv")
        assert np.array_equal(df[["slot","uid"]],prev[["slot","uid"]])
        np.testing.assert_allclose(df.refresh-df.wait,prev.gain,atol=1e-8)
    else:
        cfg=load_config()
        cfg["simulation"]["service_s"]=SLOTS*PERIOD
        world=World(cfg,"network/refresh_minus_wait/expired_training",m,n=N)
        bundle,radio=objects(world)
        rng=np.random.default_rng(19270+m)
        order,offset=order_and_offsets(rng)
        for i in range(N):
            tr=Tracker(cfg)
            for t in np.arange(0.,180.,2.)+i*.125+COST:
                tr.observe(world.observe(i,t))
                if t>=108:
                    tr.issue()
                    tr.snapshots.clear()
            st=publish(tr.evidence.archive[-1],bundle,radio,cfg)
            # Independent alternative waits from the same forecast, stratified by age.
            age_ranges=[(9.7,10.),(10.,11.),(11.,13.),(13.,16.),(16.,21.)]
            for k,(lo,hi) in enumerate(age_ranges):
                t=st.package.origin+rng.uniform(lo,hi)
                row,_,_=label(world,bundle,radio,tr,st,t,i,order[i],offset[i],k,tag)
                rows.append(row)
        df=pd.DataFrame(rows)
    df.to_csv(DEST/f"{tag}_{m}.csv",index=False)
    return tag,m,len(df)

def fit():
    data=pd.concat([pd.read_csv(p) for tag in ["replay","extra"] for p in sorted(DEST.glob(tag+"_*.csv"))])
    fields=[f"x{i}" for i in range(len(data.filter(regex=r"^x\d+$").columns))]
    X=data[fields].to_numpy()
    def forest():
        return RandomForestRegressor(n_estimators=80,min_samples_leaf=9,max_features=.85,
                                     random_state=7,n_jobs=1)
    refresh=forest().fit(X,data.refresh)
    valid=data.valid>0
    # Model the per-available-quantum mean, then multiply by known available
    # quanta at decision time. A pair straddling ten seconds only loses one.
    wait=forest().fit(X[valid],data.wait[valid]/data.valid[valid])
    with (DEST/"heads.pkl").open("wb") as f:
        pickle.dump((refresh,wait),f)
    print("train rows",len(data),"expired",sum(data.x0>10),"partial",sum(data.valid==1),flush=True)

def compare_mission(m):
    world,initial=load_pickle(OUT/f"test_{m}.pkl")
    bundle,radio=objects(world)
    init=initial_states(initial,bundle,radio,world.cfg)
    old_head=load_pickle(OUT/"rankers.pkl")[1]
    refresh,wait=load_pickle(DEST/"heads.pkl")
    order,offset=order_and_offsets(np.random.default_rng(8100+m))
    records,cf=[],[]
    for policy in ["edf","bits","split"]:
        trackers,states=copy.deepcopy((initial,init))
        for k in range(SLOTS):
            t=180+k*PERIOD
            X=np.array([features(st,t,dt) for st,dt in zip(states,offset)])
            age=X[:,0]
            nv=np.array([valid_count(st,t,order[i],world.cfg["simulation"]["beam_switch_s"]) for i,st in enumerate(states)])
            oldscore=old_head.predict(X)
            freshscore=refresh.predict(X)
            waitscore=wait.predict(X)*nv
            score=freshscore-waitscore
            assert np.all(waitscore[age>10]==0.)
            ranks={"edf":X[:,1],"bits":oldscore,"split":score}
            picked=int(np.lexsort((np.arange(N),-age,-ranks[policy]))[0])
            # After ranking, fork every expired UAV on both bit policies' own histories.
            # A fork changes only the scored UAV, and never feeds its observation back.
            audit=set(np.flatnonzero(age>10)) if policy in ("bits","split") else set()
            outcomes=np.stack([slot_trace(world,radio,i,st,t,order[i]) for i,st in enumerate(states)])
            np.testing.assert_array_equal(nv, ROUNDS-outcomes[:,2])
            for i in sorted(audit|{picked}):
                updated,newstate=new_observation(trackers[i],i,t,world,bundle,radio)
                active=slot_trace(world,radio,i,newstate,t,order[i])
                passive=outcomes[i].copy()
                if i in audit or (policy=="bits" and k==30 and m==4):
                    cf.append(dict(policy=policy,mission=m,slot=k,uid=i,t=t,age=age[i],
                                   picked=int(i==picked),oldscore=oldscore[i],refresh_est=freshscore[i],
                                   wait_est=waitscore[i],newscore=score[i],active=active[0],passive=passive[0],
                                   gain=active[0]-passive[0]))
                if i==picked:
                    trackers[i],states[i]=updated,newstate
                    outcomes[i]=active
                    nv[i]=ROUNDS-int(active[2])
            for i,(bits,loss,missing) in enumerate(outcomes):
                records.append(dict(policy=policy,mission=m,slot=k,t=t,uid=i,observed=int(i==picked),
                                    age_before=age[i],Mbit=bits,over3=loss,unavailable=missing,
                                    expired=int(sum(t+COST+(order[i]+N*np.arange(ROUNDS))*QUANTUM+
                                        (QUANTUM+world.cfg["simulation"]["beam_switch_s"])/2 >
                                        states[i].package.origin+10+1e-9))))
    df=pd.DataFrame(records)
    legacy=pd.read_csv(OUT/f"test_{m}_ledger.csv")
    key=["policy","slot","uid"]
    a=df[df.policy!="split"].sort_values(key).reset_index(drop=True)
    b=legacy[legacy.policy.isin(["edf","bits"])].sort_values(key).reset_index(drop=True)
    assert np.array_equal(a[key],b[key])
    np.testing.assert_allclose(a[["observed","t","Mbit","over3","unavailable"]],
                               b[["observed","t","Mbit","over3","unavailable"]],rtol=0,atol=1e-8)
    assert len(df)==3*N*SLOTS and (df.groupby(["policy","slot"])["observed"].sum()==1).all()
    assert (df.expired<=df.unavailable).all()
    df.to_csv(DEST/f"network_{m}.csv",index=False)
    pd.DataFrame(cf).to_csv(DEST/f"audit_{m}.csv",index=False)
    return m

if __name__=="__main__":
    DEST.mkdir(parents=True,exist_ok=True)
    stage=sys.argv[1]
    if stage=="train":
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs=[pool.submit(train_part,(tag,m)) for tag in ["extra","replay"] for m in range(4)]
            for job in as_completed(jobs):
                print("training data",job.result(),flush=True)
        fit()
    elif stage=="test":
        with ProcessPoolExecutor(max_workers=4) as pool:
            for m in pool.map(compare_mission,range(6)):
                print("network",m,flush=True)
    else:
        raise ValueError(stage)
