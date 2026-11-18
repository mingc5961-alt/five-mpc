from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import dump_json
from .dataset import center_metrics
from .validity import interval_errors

def offline_report(test,bundle,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(center_metrics(test,bundle['methods']));df.to_csv(out/'centers_raw.csv',index=False)
    summary=df.groupby(['method','group','horizon'])[['ADE_m','FDE_m','angular_RMSE_deg','angular_p95_deg']].mean().reset_index();summary.to_csv(out/'centers_summary.csv',index=False)
    rows=[]
    for center in ['RC','CR']:
        if center not in bundle['heads']:continue
        heads={'full':bundle['heads'][center],**{k:v for k,v in bundle['controls'][center].items() if k in ['global','local','no_archive_fixed_center']}}
        for name,head in heads.items():
            for a in test:
                err=interval_errors(a['package'],center,a['truth_h'],a['truth']);env=head.predict(a['package'],a['features'][center])
                ids=[0,1,2,4,6,10,16,20];qenv=env[ids]
                rows.append({'task':a['task'],'origin':a['package'].origin,'center':center,'head':name,'covered':np.all(err<=qenv),'mean_halfwidth_deg':np.rad2deg(qenv).mean()})
    v=pd.DataFrame(rows);v.to_csv(out/'fixed_center_validity.csv',index=False)
    if len(v):v.groupby(['center','head'])[['covered','mean_halfwidth_deg']].mean().to_csv(out/'fixed_center_summary.csv')
    return summary

def paired_bootstrap(a,b,seed=271828,reps=2000):
    d=np.asarray(a)-np.asarray(b);r=np.random.default_rng(seed);means=np.mean(d[r.integers(0,len(d),(reps,len(d)))],axis=1)
    return {'difference':float(d.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'paired_tasks':len(d)}

def summarize_network(root):
    paths=list(Path(root).rglob('metrics.json')); rows=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    if not rows:return []
    df=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]);df.to_csv(Path(root)/'network_summary.csv',index=False)
    settings=[c for c in ['N','sensing_budget','structurally_reusable_fraction','warmup_s','service_s','head','seed','control','scheduler','action_mode'] if c in df.columns]
    comparisons=[]
    for key,group in df.groupby(settings,dropna=False):
        if 'RC' not in set(group['method']):continue
        condition={k:(None if pd.isna(v) else v) for k,v in zip(settings,key if isinstance(key,tuple) else (key,))}
        for metric in ['net_mbps','sensing_fraction','outage_data_fraction','worst10_mbps']:
            pivot=group.pivot_table(index='task',columns='method',values=metric,aggfunc='first')
            for baseline in [m for m in pivot.columns if m!='RC']:
                pair=pivot[['RC',baseline]].dropna()
                if len(pair)<2:continue
                comparisons.append({'settings':condition,'comparison':f'RC minus {baseline}','metric':metric,**paired_bootstrap(pair['RC'],pair[baseline])})
    dump_json(Path(root)/'paired_statistics.json',comparisons)
    return df
