"""Shared five-center experiment, with matching downstream diagnostic channels."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from .training import fit_oof_gates, fit_scale, train_action_head
from .validity import QUERY_H, GlobalEnvelope, interval_errors
from .radio import Radio,array_gain
from .geometry import angles,angle_error
from .scheduler import frontier
from .dataset import center_metrics
from .evaluation import paired_bootstrap
from .config import dump_json

CENTERS=('I','P','RC','C','CR_raw')

def train_five(train,val,cal,test,cfg,out,kind='lightgbm',device='cpu',seed=7,extras=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups=[{a['task'] for a in split} for split in [train,val,cal,test]]
    if sum(map(len,groups))!=len(set.union(*groups)) or len(cal)!=len(groups[2]):
        raise ValueError('Independent route splits and one calibration anchor per route required')
    train,gates=fit_oof_gates(train,val,cal,test,seed,experts=(('RC','P'),))
    heads={k:fit_scale(train,val,cal,k,kind,device,seed,'local') for k in CENTERS}
    controls={'RC':{'global':GlobalEnvelope().fit(train,'RC').calibrate(cal,'RC')}}
    if extras:
        controls['RC']['full']=fit_scale(train,val,cal,'RC',kind,device,seed,'full')
        controls['RC']['duration']=train_action_head(train,val,cal,'RC',cfg,kind,device,seed,'duration')
    bundle=dict(heads=heads,gates=gates,controls=controls,kind=kind,seed=seed,config=cfg,
                methods=CENTERS,counts=list(map(len,groups)),fixed=None)
    with (out/'bundle.pkl').open('wb') as f:pickle.dump(bundle,f)
    return bundle

def evaluate_anchors(anchors,bundle,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    radio=Radio(bundle['config']);gate_rows=[];deadline=[]
    for a in anchors:
        pkg=a['package'];event=a.get('event','unlabelled')
        for method in CENTERS:
            times=a['truth_h'];pos=np.array([pkg.at(method,t) for t in times]);actual=angles(a['truth'])
            loss=-10*np.log10(array_gain(angles(pos),actual,(32,32))/1024)
            candidates=bundle['config']['simulation']['intervals_s']
            oracle=max((h for h in candidates if (loss[times<=h]<=3).all()),default=0.)
            variants={'common_local':bundle['heads'][method]}
            if method=='RC':
                variants.update({k:v for k,v in bundle['controls']['RC'].items() if k in ('global','full')})
                if 'duration' in bundle['controls']['RC']:variants['duration']=bundle['heads']['RC']
            for variant,head in variants.items():
                f=a['features'][method];env=head.predict(pkg,f);direct=bundle['controls']['RC'].get('duration') if variant=='duration' else None
                fs=frontier(pkg,method,env,radio,bundle['config'],direct,f)
                certified=max((x.interval for x in fs if x.pair==(32,32)),default=0.)
                cover=None if direct else bool(np.all(interval_errors(pkg,method,times,a['truth'])<=env[(QUERY_H*2).astype(int)]))
                deadline.append(dict(task=a['task'],origin=pkg.origin,method=method,head=variant,oracle_s=oracle,approved_s=certified,
                               issued=int(certified>0),early_failure=int(certified>0 and np.any(loss[times<=certified]>3)),covered=cover,
                               mean_halfwidth_deg=None if direct else np.rad2deg(env).mean()))
        for h in (1,2,5,10):
            j=int(h*2);truth=np.array([np.interp(h,a['truth_h'],x) for x in a['truth'].T])
            row=dict(task=a['task'],group=a['group'],origin=pkg.origin,event=event,horizon=h,
                     weight=pkg.diagnostics['gate_weights']['RC'][j],support=pkg.diagnostics['support'][j])
            for name in CENTERS:
                pred=pkg.at(name,h);row[name+'_m']=np.linalg.norm(pred-truth)
                row[name+'_deg']=np.linalg.norm(np.rad2deg(angle_error(angles(pred),angles(truth))))
            gate_rows.append(row)
    df=pd.DataFrame(center_metrics(anchors,CENTERS));df.to_csv(out/'forecast.csv',index=False)
    paired=[]
    by_route=df.groupby(['task','horizon','method'])[['ADE_m','FDE_m','angular_RMSE_deg','angular_p95_deg']].mean().reset_index()
    for horizon,group in by_route.groupby('horizon'):
        for metric in ('ADE_m','FDE_m','angular_RMSE_deg','angular_p95_deg'):
            table=group.pivot(index='task',columns='method',values=metric)
            for other in CENTERS:
                if other!='RC':
                    paired.append(dict(horizon=horizon,metric=metric,comparison='RC minus '+other,
                                **paired_bootstrap(table['RC'],table[other])))
    pd.DataFrame(paired).to_csv(out/'paired_forecast.csv',index=False)
    pd.DataFrame(gate_rows).to_csv(out/'gate_events.csv',index=False)
    pd.DataFrame(deadline).to_csv(out/'deadlines.csv',index=False)
    return df

