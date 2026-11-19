"""Paired service-only cost/budget experiment starting from identical dense observations."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import ch6_run as runner
from ch6_dispatch import SupervisorLock

PROJECT = Path(__file__).resolve().parents[1]
ARCH = Path('/data/chen/routeclock_archives/paired600')
MAIN = ARCH/'runs/paired600'
EXT = Path('/data/chen/routeclock_runs/paired600_pressure12_4')
ROOT = Path('/data/chen/routeclock_runs/fixed_history_20261119')
PYTHON = '/data/chen/server_assessment/venv/bin/python'
SCRIPT = Path(__file__).resolve()


def read(p):
    return json.loads(p.read_text())


def network():
    import ch6_experiment as experiment
    import numpy as np
    simulate = experiment.simulate
    def recorded(cfg, world, bundle, method, adapter, out, **kwargs):
        histories = [t.history for t in kwargs['service_trackers']]
        digest = hashlib.sha256()
        counts = []
        for history in histories:
            counts.append(len(history))
            digest.update(np.asarray([len(history)],dtype='<i4').tobytes())
            for o in history:
                for value in (o.t,o.raw,o.position,o.covariance,o.radial_velocity,o.radial_std):
                    digest.update(np.asarray(value,dtype='<f8').tobytes())
        if sum(counts)!=2872 or min(counts)!=179 or max(counts)!=180:
            raise ValueError(f'Unexpected dense history: {counts}')
        runner.write(out/'warm_history.json',dict(world=world.mission,counts=counts,sha256=digest.hexdigest(),
                     warm_cost_ms=3.1,warm_budget_pct=5.0,warm_end_s=180))
        return simulate(cfg,world,bundle,method,adapter,out,**kwargs)
    experiment.simulate = recorded
    experiment.main()


def option(argv,key,value):
    argv[argv.index(key)+1] = str(value)


def matrix():
    rows=[]; reused=[]; common=[]
    model=MAIN/'models/frozen/bundle.pkl'
    assert runner.digest(model)==read(MAIN/'models/frozen/selection.json')['sha256']==runner.digest(EXT/'models/frozen/bundle.pkl')
    for base in (MAIN,EXT):
        for ref in read(base/'manifest.json')['confirmation']:
            origin=base/'test'/Path(ref['out']).parent.name/Path(ref['out']).name
            row=dict(ref,source=str(origin),policy='original')
            baseline=(row['cost_ms'],row['budget_pct'])==(3.1,5.)
            if baseline:reused.append(dict(row,out=str(origin)))
            kinds=[] if baseline else ['paired']
            if row['method']=='F' and row['budget_pct']>=2.5:
                kinds.append('fixed_action')
            if baseline and row['world']==0 and row['method']=='I':kinds.append('baseline_check')
            for kind in kinds:
                out=ROOT/kind/origin.parent.name/origin.name.replace('lifecycle','fixed')
                argv=ref['argv'].copy()
                argv[0:2]=[PYTHON,str(SCRIPT)]
                for key,value in [('--warm','fixed'),('--out',out),('--bundle',model)]:
                    option(argv,key,value)
                if row['method']=='F':
                    # Transfer the archived action unchanged. Keep its original provenance.
                    source=MAIN/'fixed'/('cost3.1_budget1.6666667' if kind=='fixed_action' else origin.parent.name)/'winner.json'
                    old=read(source)
                    policy=ROOT/'policies'/kind/origin.parent.name/'policy.json'
                    policy.parent.mkdir(parents=True,exist_ok=True)
                    payload=dict(warm='fixed',budget_pct=row['budget_pct'],cost_ms=row['cost_ms'],smoke=False,
                                 pair=old['pair'],interval=old['interval'],source=str(source),
                                 source_sha256=runner.digest(source),purpose='transfer frozen action')
                    if policy.exists():assert read(policy)==payload
                    else:runner.write(policy,payload)
                    option(argv,'--winner',policy)
                record=dict(row,argv=argv,out=str(out),warm='fixed',policy=kind)
                (rows if kind=='paired' else common).append(record)
    assert (len(rows),len(reused),len(common))==(520,140,61)
    # Keep related cells near each other so their read-only warm states stay cached.
    jobs=sorted(common+rows,key=lambda r:(r['world'],r['policy']!='baseline_check',r['method']))
    plan=dict(new=jobs,reused=reused)
    mp=ROOT/'manifest.json'
    if mp.exists():assert read(mp)==plan
    else:runner.write(mp,plan)
    return jobs


def main():
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    lock=SupervisorLock(ROOT)
    runner.ROOT=ROOT
    jobs=matrix()
    runner.write(ROOT/'protocol.json',dict(frozen=runner.digest(MAIN/'models/frozen/bundle.pkl'),
        worlds=30,new_policy_cells=520,reused_base=140,common_fixed_action_extra=60,baseline_replay=1,
        primary=['within-method service-only cost and budget curves','within-world dense minus lifecycle',
                 'change in RC minus I from 9.3 to 12.4 ms','three schedulers and 10-world Chronos',
                 'transferred F action against common 32x32/3s F'],
        warm='3.1 ms / 5%, same observations including timestamps and measurements',
        inference='2000 within-world paired bootstrap, descriptive pointwise intervals'))
    n=runner.execute('fixed-history',jobs,lock)
    runner.write(ROOT/'status.json',dict(stage='simulation_complete',completed=n,total=len(jobs),running=[],failed=[],pending=0,at=time.time()))


if __name__=='__main__':
    if sys.argv[1:2]==['network']:network()
    else:main()
