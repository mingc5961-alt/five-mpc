"""Fixed-clock closed loops and one-step counterfactuals in identical radio worlds."""
from prepare import OUT, ROOT, N, ROUNDS, COST, QUANTUM, PERIOD, SLOTS
import argparse
import copy
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from routeclock.geometry import angles
from routeclock.predictors import apply_gate
from routeclock.radio import Radio, array_gain
from routeclock.scheduler import frontier, choose_actions

@dataclass
class Issued:
    package: object
    tube: np.ndarray
    action: object

def publish(pkg, bundle, radio, cfg):
    apply_gate(pkg, bundle["gates"]["RC"], pkg.features, "RC", "P")
    tube = bundle["heads"]["RC"].predict(pkg, pkg.features)
    fs = frontier(pkg, "RC", tube, radio, cfg)
    action = choose_actions([fs], COST, cfg["simulation"]["sensing_budget"], mode="edf")[0]
    return Issued(pkg, tube, action)

def new_observation(tracker, uid, t, world, bundle, radio):
    tr = copy.deepcopy(tracker)
    tr.observe(world.observe(uid, t + COST))
    pkg = tr.issue()
    tr.snapshots.clear()
    return tr, publish(pkg, bundle, radio, world.cfg)

def features(state, t, offset):
    pkg, act = state.package, state.action
    age = t - pkg.origin
    h = np.clip(age + offset, 0., 10.)
    j = int(round(h * 2))
    f = pkg.features[j]
    p = pkg.at("RC", h)
    az, el = angles(p)
    due = act.interval if act else 0.
    pair = act.pair if act else (0,0)
    return [age, age - due, *pair, float(act is None), float(age + offset > 10),
            *state.tube[j], *state.tube[1], np.linalg.norm(p), np.sin(az), np.cos(az),
            el, *f[[1,2,3,4,5,6,7,8,9,12,16,18]], offset]

def slot_trace(world, radio, uid, issued, t, order):
    act, pkg = issued.action, issued.package
    switch = world.cfg["simulation"]["beam_switch_s"]
    duration = QUANTUM - switch
    times = t + COST + (order + N*np.arange(ROUNDS))*QUANTUM + (QUANTUM + switch)/2
    valid = times <= pkg.origin + 10. + 1e-9
    if act is None or not valid.any():
        return np.array([0., 0., float(ROUNDS)])
    mid = times[valid]
    p = np.stack([pkg.at("RC", ti-pkg.origin) for ti in mid])
    direction = angles(p)
    truth = np.stack([world.truth(uid, ti)[0] for ti in mid])
    shadow = np.array([world.shadow(uid, ti) for ti in mid])
    snr = radio.snr(truth, direction, act.pair, shadow)
    predicted_snr = radio.snr(p, direction, act.pair)
    mcs = np.array([radio.mcs(x) for x in predicted_snr])
    rates, _ = radio.goodput(snr, mcs)
    loss = 10*np.log10(np.prod(act.pair)/array_gain(direction, angles(truth), act.pair))
    return np.array([sum(rates)*duration/1e6, float(sum(loss > 3.)), float(sum(~valid))])

def load_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)

def run_world(key):
    split, mission = key
    world, initial = load_pickle(OUT/f"{split}_{mission}.pkl")
    bundle = load_pickle(ROOT/"runs/five_pilot/results/bundle.pkl")
    radio = Radio(world.cfg)
    init_states = [publish(tr.evidence.archive[-1], bundle, radio, world.cfg) for tr in initial]
    rng = np.random.default_rng(8100 + mission)
    # Rotate TDMA identities independently of route type. All policies get the same data slots.
    order = np.argsort(rng.permutation(N))
    train = split == "train"
    heads = None if train else load_pickle(OUT/"rankers.pkl")
    frames, decisions, rows = [], [], []
    for policy in (["explore"] if train else ["edf", "risk", "bits"]):
        trackers, states = copy.deepcopy((initial, init_states))
        for k in range(SLOTS):
            t = 180 + k*PERIOD
            ages = np.array([t-st.package.origin for st in states])
            offsets = COST + (order + N*(ROUNDS-1)/2 + .5)*QUANTUM
            X = np.array([features(st,t,off) for st,off in zip(states,offsets)])
            due_score = X[:,1]
            if train:
                # Random, oldest, and uniformly sampled alternatives cover both low and high age.
                picked = int(rng.integers(N)) if rng.random() < .65 else int(np.argmax(ages))
                candidates = sorted(set([picked, int(np.argmax(ages)), *rng.choice(N, 2, replace=False)]))
            else:
                def top(scores):
                    return int(np.lexsort((np.arange(N), -ages, -scores))[0])
                risk_scores, bit_scores = (head.predict(X) for head in heads)
                choices = dict(edf=top(due_score), risk=top(risk_scores), bits=top(bit_scores))
                picked = choices[policy]
                # On both decision histories fork the two actions before observing either UAV.
                candidates = sorted({picked, choices["risk"], choices["bits"]}) if policy != "edf" else [picked]
            old = np.stack([slot_trace(world,radio,i,st,t,order[i]) for i,st in enumerate(states)])
            delta = {}
            for uid in candidates:
                updated, fresh = new_observation(trackers[uid], uid,t,world,bundle,radio)
                outcome = slot_trace(world,radio,uid,fresh,t,order[uid])
                delta[uid] = outcome - old[uid]
                if train:
                    rows.append([mission,k,uid,*X[uid],(old[uid,1]+old[uid,2])/ROUNDS,delta[uid][0]])
                if uid == picked:
                    next_tracker, next_state = updated, fresh
            if not train:
                if policy != "edf":
                    r, b = choices["risk"], choices["bits"]
                    decisions.append(dict(mission=mission, history=policy, slot=k, t=t, risk_uid=r,
                        bits_uid=b, risk_group=world.groups[r], bits_group=world.groups[b],
                        risk_pick_refresh_Mbit=delta[r][0], bits_pick_refresh_Mbit=delta[b][0],
                        extra_Mbit=delta[b][0]-delta[r][0], risk_r=float(risk_scores[r]),
                        risk_b=float(risk_scores[b]), predicted_gain_b=bit_scores[b], predicted_gain_r=bit_scores[r],
                        age_r=ages[r], age_b=ages[b], old_over3_r=old[r,1], old_over3_b=old[b,1],
                        old_unavailable_r=old[r,2], old_unavailable_b=old[b,2]))
                outcome = old.copy()
                outcome[picked] += delta[picked]
                for i,(bits,failed,idle) in enumerate(outcome):
                    frames.append(dict(mission=mission, policy=policy, slot=k, t=t, uid=i,
                        group=world.groups[i], observed=int(i==picked), Mbit=bits, over3=failed,
                        unavailable=idle, quanta=ROUNDS))
            trackers[picked], states[picked] = next_tracker, next_state
    tag = f"{split}_{mission}"
    if train:
        pd.DataFrame(rows,columns=["mission","slot","uid",*[f"x{i}" for i in range(X.shape[1])],"risk","gain"]).to_csv(OUT/f"{tag}_train.csv",index=False)
    else:
        pd.DataFrame(frames).to_csv(OUT/f"{tag}_ledger.csv",index=False)
        pd.DataFrame(decisions).to_csv(OUT/f"{tag}_forks.csv",index=False)
    return key

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train","test"])
    args = parser.parse_args()
    train = args.stage == "train"
    with ProcessPoolExecutor(max_workers=4) as pool:
        for key in pool.map(run_world, [(args.stage,i) for i in range(4 if train else 6)]):
            print("finished",key,flush=True)
    if train:
        df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("train_*_train.csv"))])
        X = df.filter(regex="^x[0-9]+$")
        models = []
        for label in ["risk","gain"]:
            head = RandomForestRegressor(n_estimators=80,min_samples_leaf=9,max_features=.85,
                                         random_state=7,n_jobs=1)
            head.fit(X.to_numpy(),df[label])
            models.append(head)
        with (OUT/"rankers.pkl").open("wb") as stream:
            pickle.dump(models,stream)
        print("training labels",len(df),"unique routes",len(df.groupby(["mission","uid"])),flush=True)
