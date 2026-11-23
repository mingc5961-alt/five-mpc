"""Score target-device timestamps after joining a manifest of all expected decisions to actual events."""
import argparse
import csv
import json
import math
import sys

KEY=('run_id','model','decision_id')
STAGES=('sample_ns','ingress_ns','start_ns','ready_ns','sent_ns','effective_ns')

def percentile(values,p):
    a=sorted(values);k=(len(a)-1)*p
    return a[int(k)]+(a[math.ceil(k)]-a[int(k)])*(k-int(k))

def score(rows,limit):
    if limit is not None and (not math.isfinite(limit) or limit<=0):
        raise ValueError('deadline-ms must be finite and positive')
    groups={};seen=set()
    for row in rows:
        key=tuple(row[k].strip() for k in KEY)
        if not all(key) or key in seen:raise ValueError(f'duplicate or missing ID {key}')
        seen.add(key)
        ts=[int(row[k]) if row[k].strip() else None for k in STAGES]
        if ts[0] is None:raise ValueError(f'missing expected sample timestamp {key}')
        valid=[t for t in ts if t is not None]
        if min(valid)<0 or any(b<a for a,b in zip(valid,valid[1:])):
            raise ValueError(f'noncausal timestamp order {key}')
        deadline=int(row['deadline_ns']) if row.get('deadline_ns','').strip() else None
        if deadline is None:
            if limit is None:raise ValueError('supply absolute deadline_ns or --deadline-ms')
            deadline=ts[0]+round(limit*1e6)
        if deadline<=ts[0]:raise ValueError(f'invalid deadline {key}')
        g=groups.setdefault(key[:2],dict(expected=0,missing_effect=0,late_effect=0,obsolete_effect=0,valid_on_time=0,latencies={}))
        g['expected']+=1
        flag=(row.get('obsolete') or '').strip().lower()
        if flag not in ('','0','false','1','true'):raise ValueError(f'invalid obsolete flag {key}')
        obsolete=flag in ('1','true')
        if ts[-1] is None:
            g['missing_effect']+=1
        else:
            g['late_effect']+=int(ts[-1]>deadline)
            g['obsolete_effect']+=int(obsolete)
            g['valid_on_time']+=int(ts[-1]<=deadline and not obsolete)
        for name,i,j in [('sample_to_effect',0,5),('sample_to_host',0,1),('host_queue',1,2),
                         ('compute_and_schedule',2,3),('host_send',3,4),('send_to_effect',4,5)]:
            if ts[i] is not None and ts[j] is not None:g['latencies'].setdefault(name,[]).append((ts[j]-ts[i])/1e6)
    if not groups:raise ValueError('no expected decisions')
    report=[]
    for (run,model),g in groups.items():
        g['on_time_pct']=100*g['valid_on_time']/g['expected']
        distributions=g.pop('latencies')
        for name,a in distributions.items():
            g[name]={'samples':len(a),'p50_ms':percentile(a,.5),'p95_ms':percentile(a,.95),
                     'p99_ms':percentile(a,.99),'max_ms':max(a)}
        report.append(dict(run_id=run,model=model,**g))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv');ap.add_argument('--deadline-ms',type=float)
    args=ap.parse_args()
    with open(args.csv,encoding='utf-8-sig',newline='') as f:result=score(csv.DictReader(f),args.deadline_ms)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
