import numpy as np
from .geometry import angles,angle_error
from .foundation import tabpfn_regressor

QUERY_H=np.array([0.,.5,1.,2.,3.,5.,8.,10.])
EPS=np.deg2rad(.001)

def finite_quantile(scores,alpha):
    scores=np.sort(np.asarray(scores)); k=int(np.ceil((len(scores)+1)*(1-alpha)))
    if k>len(scores):raise ValueError(f'At least {int(np.ceil(1/alpha)-1)} independent calibration tasks required; got {len(scores)}')
    return float(scores[k-1])

def feature_rows(features,hs,queries=QUERY_H,mode='full'):
    # Shared diagnostics, numeric h/10, two axis one-hot indicators.
    rows=[]
    for h in queries:
        f=np.array([np.interp(h,hs,features[:,j]) for j in range(features.shape[1])])
        if mode=='local': f=np.r_[f[0],f[16:28]]
        elif mode=='no_archive':f=f.copy(); f[[6,7,8,9,10,11]]=0; f[28:56]=0
        elif mode=='no_route': f=np.r_[f[0],f[16:28]]
        for axis in [0,1]:rows.append(np.r_[f,h/10,float(axis==0),float(axis==1)])
    return np.array(rows)

def interval_errors(package,center,truth_times,truth):
    pred=np.stack([package.at(center,float(h)) for h in truth_times]); errors=np.abs(angle_error(angles(truth),angles(pred)))
    running=np.maximum.accumulate(errors,axis=0)
    return np.array([running[min(np.searchsorted(truth_times,h,side='right')-1,len(running)-1)] for h in QUERY_H])

class Regressor:
    _active_tabpfn=None
    def __init__(self,kind='lightgbm',device='cpu',seed=7,params=None):
        self.kind=kind; self.device=device; self.seed=seed; self.params=params or {}
    def fit(self,X,y,Xv=None,yv=None):
        if self.kind=='tabpfn':
            self.train_X=np.asarray(X).copy();self.train_y=np.asarray(y).copy()
            self._activate()
            self.model=tabpfn_regressor(self.device,self.seed); self.model.fit(X,y)
        elif self.kind=='lightgbm':
            import lightgbm as lgb
            self.model=lgb.LGBMRegressor(n_estimators=500,num_leaves=self.params.get('num_leaves',15),min_child_samples=self.params.get('min_child_samples',20),verbosity=-1,n_jobs=4,random_state=self.seed)
            kwargs={}
            if Xv is not None:kwargs={'eval_X':Xv,'eval_y':yv,'callbacks':[lgb.early_stopping(30,verbose=False)]}
            self.model.fit(X,y,**kwargs)
        else:raise ValueError(f'Unknown real model {self.kind}')
        return self
    def _activate(self):
        previous=Regressor._active_tabpfn
        if previous is not None and previous is not self:previous.model=None
        Regressor._active_tabpfn=self
    def predict(self,X):
        if self.kind=='tabpfn':
            self._activate()
            if getattr(self,'model',None) is None:
                self.model=tabpfn_regressor(self.device,self.seed);self.model.fit(self.train_X,self.train_y)
        return np.asarray(self.model.predict(X))
    def __getstate__(self):
        state=self.__dict__.copy()
        if self.kind=='tabpfn':state['model']=None
        return state

class ScaleHead:
    def __init__(self,kind='lightgbm',device='cpu',seed=7,mode='full',params=None):
        self.reg=Regressor(kind,device,seed,params); self.mode=mode; self.factor=None
    def fit(self,X,y,Xv=None,yv=None):self.reg.fit(X,np.log(EPS+y),Xv,np.log(EPS+yv) if yv is not None else None);return self
    def raw(self,package,features):
        rows=feature_rows(features,package.hs,mode=self.mode); pred=self.reg.predict(rows)
        return np.maximum.accumulate(np.maximum(np.exp(np.clip(pred,-30,5))-EPS,EPS).reshape(-1,2),axis=0)
    def calibrate(self,anchors,center):
        scores=[]
        for a in anchors:
            err=interval_errors(a['package'],center,a['truth_h'],a['truth'])
            scores.append(float(np.max(err/self.raw(a['package'],a['features'][center]))))
        self.factor=finite_quantile(scores,.1); self.calibration_scores=scores; return self
    def predict(self,package,features):
        if self.factor is None:raise RuntimeError('Scale head has not been calibrated')
        values=self.raw(package,features)*self.factor
        idx=np.searchsorted(QUERY_H,package.hs,side='left')
        return values[idx]

class GlobalEnvelope:
    def fit(self,anchors,center):
        errors=np.array([interval_errors(a['package'],center,a['truth_h'],a['truth']) for a in anchors]); self.scale=np.maximum(np.median(errors,axis=0),EPS); return self
    def calibrate(self,anchors,center):
        self.factor=finite_quantile([np.max(interval_errors(a['package'],center,a['truth_h'],a['truth'])/self.scale) for a in anchors],.1);return self
    def predict(self,package,features):return self.scale[np.searchsorted(QUERY_H,package.hs)]*self.factor

class ActionHead:
    """Direct first-loss duration or future communication-SNR lower bound."""
    def __init__(self,kind='lightgbm',device='cpu',seed=7,target='duration'):
        self.reg=Regressor(kind,device,seed); self.target=target;self.correction=None
    def rows(self,features,hs,pair,queries):
        rows=[]
        for h in queries:
            f=np.array([np.interp(h,hs,features[:,j]) for j in range(features.shape[1])])
            rows.append(np.r_[f,pair[0]/32,pair[1]/32,h/10])
        return np.array(rows)
    def fit(self,X,y,Xv=None,yv=None):self.reg.fit(X,y,Xv,yv);return self
    def calibrate(self,scores):self.correction=finite_quantile(scores,.1);return self
    def lower(self,X):
        if self.correction is None:raise RuntimeError('Action head uncalibrated')
        return self.reg.predict(X)-self.correction
