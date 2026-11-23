"""Manifest for the 30-world confirmation and its independent validation jobs."""
import argparse
import csv
import shlex
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
BASE=(3.1,5.)
BUDGET_POINTS=(BASE,(3.1,2.5),(3.1,5./3))
PRESSURE=((6.2,5.),(9.3,5.))
ENDS=(BASE,BUDGET_POINTS[-1],PRESSURE[-1])


def case_name(point):
    c,rho=point
    return f'cost{c:g}_budget{rho:.8g}'


def jobs(stage,root,checkpoint):
    rows={}
    split='test' if stage=='confirmation' else 'validation'
    worlds=range(30) if split=='test' else range(1,21)
    models={'G0':PROJECT/'results/g0_local_delta/bundle.pkl',
            'local':PROJECT/'results/g1_recent_local_delta/bundle.pkl',
            'full':root/'models/g1_full/bundle.pkl','frozen':root/'models/frozen/bundle.pkl'}
    def add(point,i,method,model='frozen',cal='deadline',warm='lifecycle',scheduler='value'):
        cost,budget=point
        name=case_name(point)
        out=root/split/name/f'{warm}_{model}_{cal}_{scheduler}_{method}_{i:02}'
        if out in rows:return
        fixed=root/'fixed'/name/'winner.json'
        cmd=[sys.executable,HERE/'ch6_experiment.py','network','--split',split,'--mission',i,
             '--cost-ms',cost,'--budget-pct',budget,'--warm',warm,'--method',method,'--bundle',models[model],
             '--use-calibration',cal,'--scheduler',scheduler,'--compact','--out',out]
        if method=='F':cmd+=['--winner',fixed]
        if method in ('C','CR_raw'):cmd+=['--checkpoint',checkpoint]
        rows[out]=dict(split=split,world=i,cost_ms=cost,budget_pct=budget,warm=warm,model=model,
                       calibration=cal,scheduler=scheduler,method=method,out=str(out),argv=list(map(str,cmd)))
    for i in worlds:
        if split=='test':
            for point in BUDGET_POINTS:
                for m in ('I','P','RC','F'):add(point,i,m)
                if i<10:
                    for m in ('C','CR_raw'):add(point,i,m)
            for point in PRESSURE:
                for m in ('I','RC'):add(point,i,m)
            for s in ('edf','uniform'):add(BUDGET_POINTS[-1],i,'RC',scheduler=s)
        else:
            for point in (*BUDGET_POINTS,*PRESSURE):
                for warm in (('lifecycle',) if point==BASE else ('lifecycle','fixed')):
                    for m in ('I','RC'):add(point,i,m,'G0','global',warm)
            for point in ENDS:
                add(point,i,'RC','G0','deadline')
                add(point,i,'RC','local','global')
                for model in ('local','full'):
                    for m in ('RC','I'):add(point,i,m,model)
            for model in ('local','full'):
                for m in ('P',*(['C','CR_raw'] if i<=10 else [])):add(PRESSURE[-1],i,m,model)
            for s in ('edf','uniform'):add(BUDGET_POINTS[-1],i,'RC','local',scheduler=s)
    if split=='test' and (len(rows)!=600 or len(set(r['world'] for r in rows.values()))!=30):
        raise ValueError('Invalid confirmation matrix')
    return list(rows.values())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['validation','confirmation'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args()
    rows=jobs(a.stage,a.root.resolve(),a.checkpoint.resolve())
    fields=[k for k in rows[0] if k!='argv']+['command']
    w=csv.DictWriter(sys.stdout,fieldnames=fields)
    w.writeheader()
    for r in rows:w.writerow({**{k:r[k] for k in fields[:-1]},'command':shlex.join(r['argv'])})


if __name__=='__main__':main()
