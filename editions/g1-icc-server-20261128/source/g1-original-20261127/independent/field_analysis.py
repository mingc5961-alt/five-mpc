"""Aggregate matched external LOS errors by flight before deriving paired intervals."""
import csv
import gzip
import json
from collections import defaultdict

import numpy as np

from history import put
from paired_grid import BASE

OUT=BASE/'field'
MODELS=('adaptive','motion','Fstar','RAFT','Chronos')
def group(rows):
    by_cell=defaultdict(list)
    for row in rows:
        if (row['label']=='same_direction_revisit' and int(row['n']) in (6,8,12,24,40)
                and float(row['age']) in (2,3,5)):
            by_cell[(row['n'],row['age'])].append([float(row[m+'_deg']) for m in MODELS])
    if not by_cell:return None
    return np.mean([np.mean(values,axis=0) for values in by_cell.values()],axis=0)

def effects(scores):
    values=np.asarray(scores)
    if len(values)==0:return dict(units=0)
    diff=values[:,1:]-values[:,0,None]
    result=dict(units=len(values),mean_degrees=dict(zip(MODELS,values.mean(axis=0))),
                gaps_vs_adaptive_deg=dict(zip(MODELS[1:],diff.mean(axis=0))))
    if len(values)>=2:
        rng=np.random.default_rng(271127)
        means=[diff[rng.integers(len(values),size=(1000,len(values)))].mean(axis=1) for _ in range(20)]
        ci=np.quantile(np.concatenate(means),[.025,.975],axis=0).T
        result['paired_ci95']={m:p.tolist() for m,p in zip(MODELS[1:],ci)}
    return result

def main():
    inputs=json.loads((OUT/'inputs.json').read_bytes())
    endpoints=[];reports={}
    for source,flights in inputs['flights'].items():
        scores=[];by_day=defaultdict(list);total=0;primary=0
        for name in flights:
            with gzip.open(OUT/'scores'/f'{source}_{name}.csv.gz','rt',newline='') as f:rows=list(csv.DictReader(f))
            receipt=json.loads((OUT/'receipts'/f'{source}_{name}.json').read_bytes())
            if len(rows)!=receipt['queries']:raise ValueError((source,name))
            total+=len(rows);primary+=receipt['primary_revisit_queries']
            value=group(rows)
            if value is None:continue
            scores.append(value);endpoints.append(dict(source=source,flight=name,**dict(zip(MODELS,value))))
            days=sorted({r['cluster'] for r in rows})
            for day in days:
                one=group(r for r in rows if r['cluster']==day)
                if one is not None:by_day[day].append(one)
        reports[source]=dict(all_flights=len(flights),all_queries=total,primary_queries=primary,
            primary_flights=effects(scores),date_sensitivity=effects([np.mean(v,axis=0) for v in by_day.values()]))
    with (OUT/'primary_by_flight.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source','flight',*MODELS]);w.writeheader();w.writerows(endpoints)
    put(OUT/'comparisons.json',dict(estimand='all matched same-direction-revisit queries at specified paid counts and ages, equally weighted available cells within flight and equally weighted flights within source; positive gap favors adaptive',sources=reports,closed_loop_endpoints=0))

if __name__=='__main__':
    main()
