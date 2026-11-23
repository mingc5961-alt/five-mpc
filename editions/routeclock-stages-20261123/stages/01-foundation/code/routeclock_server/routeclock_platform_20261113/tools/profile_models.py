"""Profile real models at exact requested context sizes; no automatic downsizing."""
import argparse,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch,psutil
from routeclock.foundation import ChronosAdapter,tabpfn_regressor
from routeclock.config import dump_json

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--model',choices=['chronos','tabpfn'],required=True);p.add_argument('--rows',type=int,default=7200);p.add_argument('--context',type=int,default=720);p.add_argument('--out',default='runs/profile.json');a=p.parse_args()
torch.set_num_threads(4);r=np.random.default_rng(7);start=time.perf_counter()
if a.model=='chronos':
    model=ChronosAdapter(a.device);x=r.normal(size=(3,a.context)).cumsum(axis=1)
    result=model.pipeline.predict_quantiles([{'target':x,'past_covariates':{'route_x':x[0],'measurement_age':np.zeros(a.context)},'future_covariates':{'route_x':np.repeat(x[0,-1],20)}}],prediction_length=20,quantile_levels=[.1,.5,.9],cross_learning=False)
    count=sum(v.numel() for v in model.pipeline.model.parameters())
else:
    os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='true';model=tabpfn_regressor(a.device);X=r.normal(size=(a.rows,63));y=X[:,0]+X[:,1]**2;model.fit(X,y);result=model.predict(r.normal(size=(16,63)));count=a.rows
if a.device.startswith('cuda'):torch.cuda.synchronize()
dump_json(a.out,{'model':a.model,'device':a.device,'count':count,'seconds':time.perf_counter()-start,'rss_gb':psutil.Process().memory_info().rss/2**30,'peak_vram_gb':torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else 0.,'rows':a.rows,'context':a.context})
