import sys,time,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from routeclock.foundation import tabpfn_regressor
from routeclock.validity import Regressor
from routeclock.config import dump_json
torch.set_num_threads(4);r=np.random.default_rng(7);X=r.normal(size=(32,63));y=X[:,0]**2+X[:,1];query=r.normal(size=(16,63))
t=time.perf_counter();model=Regressor('tabpfn','cpu',7).fit(X,y);pred=model.predict(query)
restored=pickle.loads(pickle.dumps(model));pred2=restored.predict(query)
assert np.isfinite(pred).all();assert np.allclose(pred,pred2,rtol=1e-5,atol=1e-5)
dump_json('runs/tabpfn_local_probe.json',{'status':'passed','checkpoint':str(restored.model.model_path),'version':'V2_5','estimators':4,'train_shape':X.shape,'query_shape':query.shape,'prediction':pred,'reload_max_abs_difference':float(np.max(abs(pred-pred2))),'seconds':time.perf_counter()-t})
print('TabPFN-2.5 actual fit, 16-query prediction and serialization passed',flush=True)
