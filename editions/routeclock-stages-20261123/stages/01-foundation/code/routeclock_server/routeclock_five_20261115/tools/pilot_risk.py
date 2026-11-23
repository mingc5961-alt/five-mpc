"""Fixed fused center; compare calibrated uncertainty estimators on exactly the same future."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pilot_histories import ROOT,OUT,configuration
from routeclock.training import fit_oof_gates,fit_scale,train_action_head
from routeclock.validity import QUERY_H,interval_errors
from routeclock.scheduler import frontier
from routeclock.radio import Radio,array_gain
from routeclock.geometry import angles
from routeclock.config import dump_json

def main():
    import torch
    torch.set_num_threads(2)
    cfg=configuration();radio=Radio(cfg)
    def split(name):
        records=[]
        for path in sorted(OUT.glob(name+'_*.pkl')):
            with path.open('rb') as f:records.extend(pickle.load(f)[0])
        return records
    def repeat(rows):return [dict(a,sample_h=h) for a in rows for h in (.5,1.,2.,3.,5.,8.,10.)]
    train,val=repeat(split('train')),repeat(split('validation'))
    cal,test=split('cal'),split('network')
    train,g=fit_oof_gates(train,val,cal,test,experts=(('RC','P'),))
    with (OUT/'results/bundle.pkl').open('rb') as f:bundle=pickle.load(f)
    np.testing.assert_allclose(g['RC'].beta,bundle['gates']['RC'].beta,atol=1e-12)
    heads={'local_lightgbm':bundle['heads']['RC'],'median_envelope':bundle['controls']['RC']['global']}
    heads['local_tabpfn']=fit_scale(train,val,cal,'RC','tabpfn','cpu',7,'local')
    heads['full_lightgbm']=fit_scale(train,val,cal,'RC','lightgbm','cpu',7,'full')
    duration=train_action_head(train,val,cal,'RC',cfg,'lightgbm','cpu',7,'duration')
    out=OUT/'results';rows=[]
    with (out/'risk_heads.pkl').open('wb') as stream:pickle.dump((heads,duration),stream)
    for a in test:
        pkg=a['package'];f=a['features']['RC'];times=a['truth_h']
        pred=np.array([pkg.at('RC',t) for t in times]);loss=-10*np.log10(array_gain(angles(pred),angles(a['truth']),(32,32))/1024)
        for name in [*heads,'full_direct_duration']:
            direct=duration if name=='full_direct_duration' else None
            head=heads['full_lightgbm'] if direct else heads[name];env=head.predict(pkg,f)
            candidates=frontier(pkg,'RC',env,radio,cfg,direct,f)
            h=max((x.interval for x in candidates if x.pair==(32,32)),default=0.)
            rows.append(dict(task=a['task'],head=name,approved_s=h,admitted=int(h>0),
                early_failure=int(h>0 and (loss[times<=h]>3).any()),
                covered=None if direct else int(np.all(interval_errors(pkg,'RC',times,a['truth'])<=env[(QUERY_H*2).astype(int)]))))
    pd.DataFrame(rows).to_csv(out/'risk_heads.csv',index=False)
if __name__=='__main__':main()

