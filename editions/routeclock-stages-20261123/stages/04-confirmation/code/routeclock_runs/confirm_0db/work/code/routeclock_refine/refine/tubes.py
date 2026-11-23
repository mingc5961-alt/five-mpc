"""Fit, calibrate and score paired tubes on one shared center per forecast."""
import argparse
import hashlib
import json
import pickle
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from routeclock.geometry import angles, angle_error
from routeclock.validity import EPS, QUERY_H, finite_quantile
from .models import Chronos,Moirai,Vector,HS
from .data import BASE
from .samples import paths,records

MODELS={'chronos':Chronos,'moirai':Moirai,'vector':Vector}
DENSE=np.arange(501)*.02
FUTURE=np.searchsorted(HS,DENSE).clip(0,20)
PREVIOUS=np.where(np.isclose(DENSE,HS[FUTURE]),FUTURE,np.maximum(0,FUTURE-1))

def features(history,center):
    """Identical observable feature contract across all three predictors."""
    o=history[-1]
    gaps=np.diff([h.t for h in history[-9:]])
    speed=(np.linalg.norm(o.position-history[-2].position)/gaps[-1]) if len(gaps) else 0.
    common=np.r_[o.position,np.diag(o.covariance),o.radial_velocity,o.radial_std,
                 len(history),gaps[-1] if len(gaps) else 0.,np.mean(gaps) if len(gaps) else 0.,speed]
    per_step=np.column_stack([HS,center-center[:1],angles(center),np.linalg.norm(center,axis=1)])
    merged=np.column_stack([per_step,np.tile(common,(len(HS),1))])
    return np.column_stack([np.repeat(merged,2,axis=0),np.tile([0,1],len(HS))])

def errors(center,truth):
    dense=np.column_stack([np.interp(DENSE,HS,v) for v in center.T])
    return abs(angle_error(angles(truth),angles(dense)))

def scores(err,raw):
    dense=np.maximum(raw[:,FUTURE],raw[:,PREVIOUS])
    normalized=np.max(err/dense,axis=2)
    return np.maximum.accumulate(normalized,axis=1)[:,np.rint(QUERY_H[1:]/.02).astype(int)]

def slim_predictions(model,files):
    pending=[];rows=[]
    def flush():
        if not pending:return
        forecasts=model.batch([h for _,_,h,_ in pending])
        for (flight,origin,h,truth),(center,scale) in zip(pending,forecasts,strict=True):
            if center.shape!=(21,3) or scale.shape!=(21,2) or not np.isfinite([*center.flat,*scale.flat]).all():
                raise ValueError('invalid model output')
            rows.append((flight,origin,features(h,center),errors(center,truth),scale,
                         hashlib.sha256(np.ascontiguousarray(center).tobytes()).hexdigest()))
        pending.clear()
    for path in files:
        for h,truth in records(path):
            pending.append((path.stem,h[-1].t,h,truth))
            if len(pending)==8: flush()
    flush()
    return tuple(np.stack(items) for items in zip(*rows))

def tree_scale(tree,X):
    return np.maximum(EPS,np.exp(np.clip(tree.predict(X.reshape(-1,X.shape[-1]),num_threads=1),-30,5)).reshape(-1,21,2)-EPS)

def fit(name,pilot):
    model=MODELS[name]()
    files={s:paths(s) for s in ['train','validation','calibration']}
    if pilot:
        files={s:rows[:min(len(rows),24)] for s,rows in files.items()}
        assert min(map(len,files.values()))>=10
    else:
        assert list(map(len,files.values()))==[1200,400,400]
        files['train']=[f for f in files['train'] if int(f.stem)%5==0]
    data={s:slim_predictions(model,rows) for s,rows in files.items()}
    _,_,xt,et,_,_=data['train'];_,_,xv,ev,_,_=data['validation']
    def xy(x,e):return x.reshape(-1,x.shape[-1]),np.log(EPS+e[:,::25].reshape(-1))
    X,y=xy(xt,et);V,w=xy(xv,ev)
    candidates=[];best=float('inf')
    for leaves,minimum in [(15,20),(15,50),(31,20),(31,50)]:
        m=LGBMRegressor(n_estimators=500,num_leaves=leaves,min_child_samples=minimum,n_jobs=2,random_state=7,verbosity=-1)
        m.fit(X,y,eval_set=[(V,w)],callbacks=[early_stopping(30,verbose=False)])
        mse=float(np.mean((m.predict(V)-w)**2));candidates.append([leaves,minimum,mse])
        if mse<best:best,tree=mse,m
    tree.booster_.params['num_threads']=1
    _,_,xc,ec,native,hashes=data['calibration']
    bundle={'tree':tree,'model':name,'factors':{},'fixed':np.maximum(EPS,np.median(et[:,::25],axis=0))}
    report={'model':name,'pilot':pilot,'routes':{k:len(v) for k,v in files.items()},'same_center_pairs':len(hashes),
            'center_digest':hashlib.sha256(''.join(hashes).encode()).hexdigest(),'lightgbm_candidates':candidates,
            'validation':{},'score_grid_s':.02,'seed':7}
    units=dict(native=native,lightgbm=tree_scale(tree,xc),fixed=np.broadcast_to(bundle['fixed'],native.shape))
    for method,raw in units.items():
        sc=scores(ec,raw)
        factors=np.maximum.accumulate([finite_quantile(v,.1) for v in sc.T])
        bundle['factors'][method]=factors
        valraw={'native':data['validation'][4], 'fixed':np.broadcast_to(bundle['fixed'],data['validation'][4].shape)}.get(method)
        if valraw is None:valraw=tree_scale(tree,xv)
        report['validation'][method]=dict(factors=factors.tolist(),prefix_coverage=np.mean(scores(ev,valraw)<=factors,axis=0).tolist())
    out=BASE/('pilot' if pilot else 'weights');out.mkdir(exist_ok=True)
    f=out/(name+'_tubes.pkl')
    with f.with_suffix('.part').open('wb') as stream:pickle.dump(bundle,stream)
    f.with_suffix('.part').replace(f)
    (out/(name+'_fit.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('model',choices=MODELS);arg.add_argument('--pilot',action='store_true')
    args=arg.parse_args();fit(args.model,args.pilot)
