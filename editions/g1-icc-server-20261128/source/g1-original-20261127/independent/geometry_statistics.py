"""Summarize paired geometry and expiry separately within each physical world."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import pandas as pd
from history import put
from paired_grid import BASE
from slots import compute_slot

OUT=BASE/'geometry_panel'
MODELS=('Acal','M','Fstar','IMM','RAFT','Chronos')
def one(job):
    with compute_slot():
        return world_statistics(*job)

def world_statistics(split,k):
    folder=OUT/split
    source=pd.read_csv(folder/'scores'/f'{k}.csv.gz')
    receipt=json.loads((folder/'receipts'/f'{k}.json').read_bytes())
    if len(source)!=receipt['queries'] or not (source.world==k).all():
        raise ValueError(f'geometry world {k}')
    destination=folder/'strata'/f'{k}.csv.gz'
    frames=[]
    metrics=[f'{m}_deg' for m in (*MODELS,'snapshot_age','snapshot_no_age')]
    metric_ops={m:['mean',lambda x:x.quantile(.9)] for m in metrics}
    data=source.copy()
    for model in MODELS:
        true=source[model+'_remaining_true'];admit=source[model+'_remaining_admitted']
        data[model+'_overadmitted_s']=(admit-true).clip(lower=0)
        data[model+'_unused_s']=(true-admit).clip(lower=0)
        data[model+'_expired']=(true<=0).astype(float)
        for suffix in ('remaining_true','remaining_admitted','censored','overadmitted_s','unused_s','expired'):
            metric_ops[model+'_'+suffix]=['mean']
    for label,sub in [('all',data),('natural',data[data.natural==1])]:
        groups=sub.groupby(['n','age','h','label'],observed=True)
        result=groups.agg(metric_ops)
        result.columns=[key+'_'+('p90' if method.startswith('<lambda') else method) for key,method in result.columns]
        result['users']=groups.u.nunique();result['queries']=groups.size()
        result['subset']=label;result['world']=k;frames.append(result.reset_index())
    dest_tmp=destination.with_suffix(f'.{os.getpid()}.tmp')
    pd.concat(frames).to_csv(dest_tmp,index=False,compression='gzip')
    dest_tmp.replace(destination)
    return k,len(source)

def summarize(split,n):
    folder=OUT/split
    (folder/'strata').mkdir(parents=True,exist_ok=True)
    start=1400 if split=='development' else 2000
    with ProcessPoolExecutor(min(12,n)) as workers:
        records=list(workers.map(one,[(split,k) for k in range(start,start+n)]))
    all_cells=pd.concat([pd.read_csv(folder/'strata'/f'{k}.csv.gz') for k,_ in records])
    strata=['subset','label','n','age','h']
    group=all_cells.groupby(strata,observed=True)
    numeric=list(all_cells.columns.difference([*strata,'world','users','queries']))
    report=group[numeric].mean()
    report['worlds']=group.world.nunique()
    report['possible_worlds']=n;report['worlds_without_cell']=n-report['worlds']
    report['users_in_worlds']=group.users.sum();report['queries']=group.queries.sum()
    report.to_csv(folder/'stratified_geometry.csv.gz',compression='gzip')
    put(folder/'stratified_summary.json',dict(worlds=n,queries=sum(c for _,c in records),
        cells=len(report),closed_loop_endpoints=0,aggregation='equal worlds; P90 is the mean of within-world cell quantiles'))

if __name__=='__main__':
    if (OUT/'development/summary.json').is_file() and not (OUT/'development/stratified_summary.json').is_file():
        summarize('development',8)
    if (OUT/'summary.json').is_file():
        summarize('confirmation',json.loads((OUT/'summary.json').read_bytes())['worlds'])
