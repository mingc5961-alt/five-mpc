"""Refit the RC external head with the same recipe, folds and features as the Chronos quantile tree."""
import csv
import pickle
import sys
from unittest.mock import patch
import numpy as np
from lightgbm import LGBMRegressor
from compare import OUT,CACHE,ROOT,network,tubes,paths,records,write,run_job
from runtime.remaining_models import ARCHIVE
from routeclock.predictors import apply_gate
from routeclock.validity import finite_quantile

def collect(split):
    cfg=pickle.loads((CACHE/'classical.pkl').read_bytes())['RC']
    files=paths(split)
    if split=='train':files=[f for f in files if int(f.stem)%5==0]
    assert len(files)==(240 if split=='train' else 400)
    xs=[];ys=[]
    for f in files:
        with (ARCHIVE/split/(f.stem+'.pkl')).open('rb') as fp: rows=pickle.load(fp)
        for r,(history,truth) in zip(rows,records(f),strict=True):
            p=r['package'];assert history[-1].t==p.origin
            p.diagnostics['recent_spacing_s']=np.diff([o.t for o in history[-9:]]).mean() if len(history)>1 else 0.
            apply_gate(p,cfg['gate'],r['features']['P'],'RC','P')
            c=p.centers['RC'];xs.append(tubes.features(history,c));ys.append(tubes.errors(c,truth))
    return np.stack(xs),np.stack(ys)
def fit():
    x,y=collect('train')
    tree=LGBMRegressor(objective='quantile',alpha=.9,n_estimators=130,num_leaves=15,n_jobs=1,random_state=7,verbosity=-1)
    tree.fit(x.reshape(-1,x.shape[-1]),y[:,::25].reshape(-1))
    metrics={}
    for split in ('calibration','validation'):
        x,y=collect(split);raw=np.maximum(tubes.EPS,tree.predict(x.reshape(-1,x.shape[-1]),num_threads=1).reshape(-1,21,2))
        s=tubes.scores(y,raw)
        if split=='calibration':factors=np.maximum.accumulate([finite_quantile(v,.1) for v in s.T])
        metrics[split]=(s<=factors).mean(0).tolist()
    folder=OUT/'RC_matched';folder.mkdir(exist_ok=True)
    (folder/'bundle.pkl').write_bytes(pickle.dumps(dict(tree=tree,factors=factors)))
    write(folder/'coverage.json',metrics)

if __name__=='__main__':
    fit()
