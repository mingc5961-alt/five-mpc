"""Develop an explicitly labelled Linux re-fit comparison; quarantine original holdouts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os
from collections import defaultdict
from pathlib import Path
from server_paths import RUN
from closed import SCALES,score,freeze,quota_period,destination

WORLDS=range(1400,1408)
MAIN=('N01','N02','N03','N04','N06','P01','P03','Q01','Q02','Q05')
# The fixed-clock motion and fixed-weight controls share the P03 purchase stream.
EXTRA=('P_M','P_Fstar')


def read(split,k,row,b,m,q=None,p=None):
    return json.loads(destination(split,k,row,b,m,q,p).read_bytes())


def anchor_freeze():
    result={}
    for b in (3,4.5,9):
        by={m:[read('development',k,'N01',b,m) for k in WORLDS] for m in SCALES}
        policy=freeze(by)
        if not policy['feasible']:
            raise ValueError('anchor grid needs additional development endpoints')
        quota,period=quota_period(by,policy,b)
        result[str(b)]=dict(K=quota,PM_ms=period,policy=policy)
    (RUN/'anchor_freeze.json').write_text(json.dumps(result,indent=2))
    return result


def params(row,anchor):
    return (anchor['K'] if row[0] in 'PQ' else None,anchor['PM_ms'] if row=='N03' else None)


def run_jobs(jobs,workers):
    with ProcessPoolExecutor(workers) as pool:
        pending={pool.submit(score,*job):job for job in jobs}
        for i,f in enumerate(as_completed(pending),1):
            f.result()
            if i%24==0 or i==len(jobs):print('endpoint',i,len(jobs),flush=True)


def develop(workers):
    anchors=anchor_freeze();a=anchors['3']
    rows=MAIN+EXTRA
    scales={row:list(SCALES) for row in rows}
    def dispatch(rows,grid):
        jobs=[('development',k,row,3,m,*params(row,a)) for row in rows for m in grid for k in WORLDS
              if not destination('development',k,row,3,m,*params(row,a)).exists()]
        run_jobs(jobs,workers)
    def policy(row):
        return freeze({m:[read('development',k,row,3,m,*params(row,a)) for k in WORLDS] for m in scales[row]})
    dispatch(rows,SCALES)
    for family in 'NPQ':
        rr=[row for row in rows if row.startswith(family)]
        if not all(policy(row)['feasible'] for row in rr):
            dispatch(rr,(.30,1.25))
            for row in rr:scales[row].extend([.30,1.25])
    policies={row:policy(row) for row in rows}
    # A changed native anchor invalidates dependent fixed-cost jobs.
    by={m:[read('development',k,'N01',3,m) for k in WORLDS] for m in scales['N01']}
    if quota_period(by,policies['N01'],3)!=(a['K'],a['PM_ms']):
        raise RuntimeError('expanded native grid changed K/PM; revise dependent manifest before confirmation')
    (RUN/'linux_freeze.json').write_text(json.dumps(dict(protocol='linux_refit_v1',anchor=a,policies=policies,
                    holdout=list(range(4000,4024)),physical_budget=3,shared_scale=.75),indent=2))
    return policies,a


def confirm(workers):
    freeze_file=json.loads((RUN/'linux_freeze.json').read_bytes());a=freeze_file['anchor']
    jobs=[]
    for row,pol in freeze_file['policies'].items():
        for m in {float(x) for x in pol['weights']}|{.75}:
            jobs.extend(('rebuild_holdout',k,row,3,m,*params(row,a)) for k in freeze_file['holdout'])
    run_jobs(jobs,workers)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('anchor','develop','confirm'))
    p.add_argument('--workers',type=int,default=24);args=p.parse_args()
    if args.action=='anchor':print(anchor_freeze())
    elif args.action=='develop':develop(args.workers)
    else:confirm(args.workers)
