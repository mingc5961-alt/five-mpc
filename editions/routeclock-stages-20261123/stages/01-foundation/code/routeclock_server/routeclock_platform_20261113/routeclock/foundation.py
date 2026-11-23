"""Real foundation model adapters. No replacement predictor on loading failure."""
import os
from pathlib import Path
import numpy as np
from .config import ROOT

def setup_cache():
    os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
    os.environ.setdefault('TABPFN_MODEL_CACHE_DIR',str(ROOT/'.cache/tabpfn'))
    os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1')

class ChronosAdapter:
    def __init__(self,device='cpu',checkpoint='amazon/chronos-2'):
        setup_cache()
        import torch
        from chronos import Chronos2Pipeline
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:source=snapshot_download(checkpoint,local_files_only=True)
        except LocalEntryNotFoundError:source=checkpoint
        self.pipeline=Chronos2Pipeline.from_pretrained(source,device_map=device)
        self.checkpoint=checkpoint
        self.pipeline.model.eval()
        for p in self.pipeline.model.parameters():p.requires_grad_(False)
    @staticmethod
    def inputs(history,snapshots,route_mean,route_std,route_support,with_route,step=.5,max_steps=720):
        origin=history[-1].t; n=min(max_steps,int((origin-history[0].t)/step)+1)
        grid=origin-np.arange(n-1,-1,-1)*step
        times=np.array([o.t for o in history]); pos=np.array([o.position for o in history]); idx=np.searchsorted(times,grid,side='right')-1
        idx=np.maximum(idx,0); age=grid-times[idx]; mask=(age<step).astype(float)
        offset=pos[-1]; target=(pos[idx]-offset).T
        past={'measurement_mask':mask.astype('float32'),'measurement_age':age.astype('float32')}; future={}
        if with_route:
            keys=['route_x','route_y','route_z','route_std_x','route_std_y','route_std_z','route_support']
            values=np.full((n,7),np.nan)
            origins=np.array([p.origin for p in snapshots]); ids=np.searchsorted(origins,grid,side='right')-1
            for j,k in enumerate(ids):
                if k<0:continue
                package=snapshots[k]; h=grid[j]-package.origin
                if h>package.hs[-1]:continue
                pp=package.at('P',h)-offset
                st=np.array([np.interp(h,package.hs,np.sqrt(np.maximum(package.covariance['P'][:,a,a],0))) for a in range(3)])
                support=np.interp(h,package.hs,package.diagnostics['support'])
                values[j]=np.r_[pp,st,support]
            fut=np.c_[route_mean[1:]-offset,route_std[1:],route_support[1:]]
            for j,key in enumerate(keys):past[key]=values[:,j].astype('float32'); future[key]=fut[:,j].astype('float32')
        result={'target':target.astype('float32'),'past_covariates':past}
        if with_route:result['future_covariates']=future
        return result,offset
    def predict(self,history,snapshots,p,pcov,support,with_route):
        inp,offset=self.inputs(history,snapshots,p,np.sqrt(np.maximum(np.diagonal(pcov,axis1=1,axis2=2),0)),support,with_route)
        qs,_=self.pipeline.predict_quantiles([inp],prediction_length=len(p)-1,quantile_levels=[.1,.5,.9],cross_learning=False,context_length=720)
        q=qs[0].detach().cpu().numpy() # variate, horizon, quantile
        center=np.vstack([history[-1].position,q[:,:,1].T+offset])
        std=(q[:,:,2]-q[:,:,0]).T/2.563103
        cov=np.array([np.diag(v*v) for v in std]); cov=np.concatenate([history[-1].covariance[None],cov])
        return center,cov

def tabpfn_regressor(device='cpu',seed=7,checkpoint=None):
    setup_cache()
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    checkpoint=checkpoint or os.environ.get('ROUTECLOCK_TABPFN_CHECKPOINT')
    if checkpoint is None:
        candidates=list((ROOT/'.cache/tabpfn').rglob('tabpfn-v2.5-regressor-v2.5_default.ckpt'))
        if candidates:checkpoint=str(candidates[0])
    # Explicit error prevents noninteractive browser login, not a model fallback.
    if checkpoint is None:
        raise FileNotFoundError('TabPFN-2.5 licensed checkpoint missing. Accept the model license and place tabpfn-v2.5-regressor-v2.5_default.ckpt in .cache/tabpfn, or set ROUTECLOCK_TABPFN_CHECKPOINT.')
    if not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2_5,device=device,n_estimators=4,fit_mode='fit_with_cache',random_state=seed,model_path=checkpoint)
