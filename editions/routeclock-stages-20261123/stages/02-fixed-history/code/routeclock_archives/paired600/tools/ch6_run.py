"""Resume a locked experiment manifest; publish only audited successful jobs."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import deque

from ch6_dispatch import Admission, SupervisorLock, gpu_task
from ch6_matrix import jobs,case_name,BUDGET_POINTS,ENDS,PROJECT,HERE

ROOT=PROJECT/'runs/paired600'
CHECKPOINT=Path('/data/chen/routeclock_server/weights/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c')
ENV={**os.environ,**dict.fromkeys(('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'),'1'),
     'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(row,folder):
    if row['method']=='fixed-grid':
        d=json.loads((folder/'winner.json').read_text())
        grid=json.loads((folder/'grid.json').read_text())
        assert (len(grid),d['validation_start'],d['budget_pct'],d['cost_ms'])==(20,1,row['budget_pct'],row['cost_ms'])
        assert all(len(g)==len(grid[0]) for g in grid)
        return
    m=json.loads((folder/'metrics.json').read_text())
    assert (m['task'],m['N'],m['method'],m['scheduler'])==(f"chapter6/{row['split']}/{row['world']}",16,row['method'],row['scheduler'])
    assert m['warmup_s']==m['service_s']==180
    assert abs(m['accounted_s']-180)<1e-6 and m['hold_segments']>0
    c=row['cost_ms']/1000
    assert math.isclose(m['sensing_budget'],row['budget_pct']/100) and math.isclose(m['observation_cost_s'],c)
    assert m['sensing_s']<=180*m['sensing_budget']+c+1e-8 and math.isfinite(m['net_mbps'])
    assert abs(sum(m['age_data_s'].values())+m['unavailable_s']-m['data_s'])<1e-6
    if row['method']=='RC':assert (folder/'gate_weights.npz').is_file()


def run_one(row):
    out=Path(row['out'])
    out.parent.mkdir(parents=True,exist_ok=True)
    # A new attempt never consumes a partial result as a completed job.
    for attempt in range(1,4):
        dest=out.with_name(out.name+f'.work.{time.time_ns()}')
        args=row['argv'].copy()
        args[args.index('--out')+1]=str(dest)
        log=ROOT/'logs'/('_'.join(out.relative_to(ROOT).parts)+'.log')
        with log.open('a') as f:
            proc=subprocess.run(args,cwd=PROJECT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
        try:
            if proc.returncode:raise RuntimeError(f'exit {proc.returncode}')
            audit(row,dest)
            a=json.loads((dest/'arguments.json').read_text());a['out']=str(out)
            write(dest/'arguments.json',a)
            write(dest/'completed.json',dict(at=time.time(),attempt=attempt))
            dest.rename(out)
            return
        except Exception as e:
            with log.open('a') as f:f.write(f'\nattempt {attempt}: {e!r}\n')
            if attempt==3:raise RuntimeError(f'{out}: {e}; {log}')


def execute(stage,rows,lock):
    pending=deque()
    completed=0
    outputs={r['out']:r for r in rows}
    for r in rows:
        d=Path(r['out'])
        if (d/'completed.json').exists():audit(r,d);completed+=1
        else:pending.append(r)
    # Drain the previous coordinator without sharing its remaining tasks.
    retained={r['out']:r for r in pending if r['out'] in lock.reserved}
    policy=Admission()
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=policy.workers) as pool:
        active={}
        while active or pending:
            reserved=lock.remaining()
            for path in list(retained):
                row=retained[path]
                if (Path(path)/'completed.json').exists():
                    audit(row,Path(path));completed+=1
                    pending.remove(row);del retained[path]
                elif path not in reserved:
                    del retained[path]
            gpu=sum(map(gpu_task,active.values()))+sum(gpu_task(outputs[p]) for p in reserved if p in outputs)
            policy.sample(gpu)
            while pending and len(active)+len(reserved)<policy.workers:
                row=policy.next_job(pending,gpu,reserved)
                if row is None:break
                active[pool.submit(run_one,row)]=row
                gpu+=gpu_task(row)
            running=sorted(reserved)+[r['out'] for r in active.values()]
            write(ROOT/'status.json',dict(stage=stage,total=len(rows),completed=completed,failed=failures,
                 running=running,pending=len(pending)-len(retained),at=time.time(),
                 resources={**policy.values,'worker_limit':policy.workers,'gpu_limit':policy.gpu_slots,'gpu_jobs':gpu}))
            if not active:
                if reserved:time.sleep(2);continue
                if pending:raise RuntimeError('Missing required model bundle')
                break
            done,_=concurrent.futures.wait(active,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                r=active.pop(fut)
                try:fut.result();completed+=1
                except Exception as e:failures.append(str(e))
    if failures:raise RuntimeError(str(failures))
    return completed


def fixed():
    rows=[]
    for cost,budget in BUDGET_POINTS:
        out=ROOT/'fixed'/case_name((cost,budget))
        args=[sys.executable,HERE/'ch6_experiment.py','select-fixed','--cost-ms',cost,'--budget-pct',budget,
              '--validation-start',1,'--out',out]
        rows.append(dict(out=str(out),method='fixed-grid',budget_pct=budget,cost_ms=cost,argv=list(map(str,args))))
    return rows


def freeze(validation):
    # Equal weight per world and operating point. All five centers have the
    # same head feature set. No confirmation-world metrics enter selection.
    means={}
    for model in ('local','full'):
        scores=[]
        for r in validation:
            point=(r['cost_ms'],r['budget_pct'])
            if (point in ENDS and r['method']=='RC' and r['model']==model and r['warm']=='lifecycle'
                and r['scheduler']=='value' and r['calibration']=='deadline'):
                scores.append(json.loads((Path(r['out'])/'metrics.json').read_text())['net_mbps'])
        assert len(scores)==60
        means[model]=sum(scores)/len(scores)
    best=max(means,key=means.get)
    src=PROJECT/'results/g1_recent_local_delta/bundle.pkl' if best=='local' else ROOT/'models/g1_full/bundle.pkl'
    dest=ROOT/'models/frozen';dest.mkdir(exist_ok=True)
    shutil.copyfile(src,dest/'bundle.tmp');(dest/'bundle.tmp').replace(dest/'bundle.pkl')
    write(dest/'selection.json',dict(selected=best,mean_net_mbps=means,sha256=digest(src)))


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='paired600',help='New campaigns are stored under /data/chen/routeclock_runs')
    parser.add_argument('--handover',type=int,help='Drain the previous coordinator PID and retain its active jobs')
    args=parser.parse_args()
    if args.run_name in ('','.','..') or Path(args.run_name).name!=args.run_name:
        parser.error('run name must be a single directory name')
    if args.run_name!='paired600' or not (ROOT/'manifest.json').exists():
        ROOT=Path('/data/chen/routeclock_runs')/args.run_name
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    (ROOT/'models').mkdir(parents=True,exist_ok=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    validation=jobs('validation',ROOT,CHECKPOINT)
    confirmation=jobs('confirmation',ROOT,CHECKPOINT)
    plan=dict(fixed=fixed(),validation=validation,confirmation=confirmation)
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Existing manifest differs')
    lock=SupervisorLock(ROOT,args.handover)
    (ROOT/'supervisor.pid').write_text(str(os.getpid()))
    write(path,plan)
    first=execute('validation',plan['fixed']+validation,lock)
    freeze(validation)
    last=execute('confirmation',confirmation,lock)
    subprocess.run([sys.executable,str(HERE/'ch6_summary.py'),str(ROOT)],env=ENV,check=True)
    write(ROOT/'status.json',dict(stage='complete',validation_including_grid=first,
                                  confirmation=last,independent_test_worlds=30,at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(repr(e),flush=True)
        raise
