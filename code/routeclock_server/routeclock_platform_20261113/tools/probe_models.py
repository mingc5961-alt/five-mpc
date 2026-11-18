"""Real checkpoint integration probe; any failed model remains explicitly failed."""
import os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TABPFN_MODEL_CACHE_DIR',str(ROOT/'.cache/tabpfn'))
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET','true')
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1')
import numpy as np
import torch
from routeclock.config import dump_json
torch.set_num_threads(4)
results={}
for name in ['chronos','tabpfn']:
    t=time.perf_counter()
    try:
        if name=='chronos':
            from routeclock.foundation import ChronosAdapter
            adapter=ChronosAdapter('cpu');model=adapter.pipeline
            target=np.stack([np.sin(np.arange(48)/8+j) for j in range(3)])
            inp={'target':target,'past_covariates':{'route_x':target[0], 'measurement_age':np.zeros(48)},'future_covariates':{'route_x':np.sin(np.arange(48,52)/8)}}
            q,mean=model.predict_quantiles([inp],prediction_length=4,quantile_levels=[.1,.5,.9],cross_learning=False)
            results[name]={'status':'passed','shape':list(q[0].shape),'values':mean[0].tolist(),'seconds':time.perf_counter()-t,'checkpoint':'amazon/chronos-2','parameters':sum(p.numel() for p in model.model.parameters())}
            del model,adapter
        else:
            from routeclock.foundation import tabpfn_regressor
            model=tabpfn_regressor('cpu',7)
            r=np.random.default_rng(7); X=r.normal(size=(32,8)); y=X[:,0]**2+X[:,1]
            model.fit(X,y); pred=model.predict(X[:4]+.1)
            results[name]={'status':'passed','values':pred.tolist(),'seconds':time.perf_counter()-t,'checkpoint':str(model.model_path),'version':'V2_5','estimators':4}
    except Exception as e:
        results[name]={'status':'failed','error':str(e),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-t}
    dump_json(ROOT/'runs/model_probe.json',results)
    print(name,results[name],flush=True)
