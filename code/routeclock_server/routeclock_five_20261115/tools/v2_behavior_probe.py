"""Early behavior diagnosis on independent single-vehicle pilot routes (no network test labels)."""
from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from pilot_histories import OUT, ROOT, configuration
from routeclock.world import World, GROUPS
from routeclock.predictors import Tracker
import hashlib


def extract(path):
    with path.open('rb') as stream:
        record=pickle.load(stream)[0][0]
    source,mission,_=record['task'].split('/')
    world=World(configuration(),'long-history-deadline-v1/'+source,int(mission),n=1,groups=[record['group']])
    assert hashlib.sha256(world.references[0].tobytes()).hexdigest()==record['route_hash']
    tracker=Tracker(configuration())
    rows=[]
    for step in range(60):
        tracker.observe(world.observe(0,step+.5))
        pkg=tracker.issue()
        if step in (29,59):
            # Only prediction-time diagnostics; 5 s is the feature query horizon.
            rows.append((record['route_hash'],record['group'],step+.5,pkg.features[10].copy()))
    return rows


if __name__=='__main__':
    out=ROOT/'runs/v2_review'
    out.mkdir(parents=True,exist_ok=True)
    # Dedicated diagnostic; calibration labels used here must not train a deployed gate.
    paths=sorted([p for prefix in ('train','validation','cal') for p in OUT.glob(prefix+'_*.pkl')])
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[r for batch in pool.map(extract,paths) for r in batch]
    output=[]
    for t in (29.5,59.5):
        subset=[r for r in rows if r[2]==t]
        hashes,labels,_,x=zip(*subset)
        assert len(hashes)==len(set(hashes))==30
        # At each age there is exactly one row per independent route.
        splits=list(StratifiedKFold(5,shuffle=True,random_state=7).split(x,labels))
        assert all(set(np.array(hashes)[a]).isdisjoint(np.array(hashes)[b]) for a,b in splits)
        pred=cross_val_predict(RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced',random_state=7,n_jobs=1),x,labels,cv=splits)
        pd.DataFrame(confusion_matrix(labels,pred,labels=GROUPS),index=GROUPS,columns=GROUPS).to_csv(out/f'confusion_{int(t)}.csv')
        output.extend(dict(route=h,time=t,label=y,prediction=p) for h,y,p in zip(hashes,labels,pred))
        print(t,balanced_accuracy_score(labels,pred),f1_score(labels,pred,average='macro'),flush=True)
    pd.DataFrame(output).to_csv(out/'early_behavior_oof.csv',index=False)
