"""Replay the previously reported decision and score all 16 alternatives."""
from refresh_minus_wait import (DEST, OUT, N, PERIOD, ROUNDS, objects, initial_states,
                                order_and_offsets, features, valid_count, new_observation,
                                slot_trace, load_pickle)
import numpy as np
import pandas as pd

def main():
    mission, slot=4,30
    world, trackers=load_pickle(OUT/f"test_{mission}.pkl")
    bundle,radio=objects(world)
    states=initial_states(trackers,bundle,radio,world.cfg)
    order,off=order_and_offsets(np.random.default_rng(8100+mission))
    prior=pd.read_csv(OUT/f"test_{mission}_ledger.csv")
    picks=prior[(prior.policy=="bits")&(prior.observed==1)].set_index("slot").uid
    for k in range(slot):
        i=int(picks[k])
        trackers[i],states[i]=new_observation(trackers[i],i,180+k*PERIOD,world,bundle,radio)
    t=180+slot*PERIOD
    X=np.stack([features(st,t,off[i]) for i,st in enumerate(states)])
    old=load_pickle(OUT/"rankers.pkl")[1].predict(X)
    assert np.argmax(old)==picks[slot]
    refresh_head,wait_head=load_pickle(DEST/"heads.pkl")
    nv=np.array([valid_count(st,t,order[i],world.cfg["simulation"]["beam_switch_s"]) for i,st in enumerate(states)])
    fresh_score=refresh_head.predict(X)
    wait_score=wait_head.predict(X)*nv
    data=[]
    for i,st in enumerate(states):
        _,fresh=new_observation(trackers[i],i,t,world,bundle,radio)
        true_refresh=slot_trace(world,radio,i,fresh,t,order[i])[0]
        true_wait=slot_trace(world,radio,i,st,t,order[i])[0]
        data.append(dict(uid=i,age=t-st.package.origin,old_score=old[i],new_score=fresh_score[i]-wait_score[i],
                         refresh_est=fresh_score[i],wait_est=wait_score[i],actual_gain=true_refresh-true_wait))
    table=pd.DataFrame(data).sort_values(["new_score","age","uid"],ascending=[False,False,True])
    table.to_csv(DEST/"example_all_candidates.csv",index=False)
    print(table.head(3).round(3).to_string(index=False))

if __name__=="__main__":
    main()
