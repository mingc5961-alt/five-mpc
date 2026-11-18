"""Declarative matrix; physical task identities are independent of methods."""
from pathlib import Path
import copy,pickle
import numpy as np
from .world import World
from .simulator import simulate
from .config import dump_json

def experiment_matrix():
    jobs=[]
    def add(block,methods,count=20,**settings):
        for mission in range(count):
            for method in methods:jobs.append({'block':block,'mission':20000+mission,'method':method,'N':16,'budget':.05,'seed':7,**settings})
    add('B',['F','I','C','CR','RC'],30)
    add('B_Z',['Z'])
    for n in [4,8,32]:add('C_N',['I','CR','RC'],N=n)
    for b in [.02,.1]:add('C_budget',['I','CR','RC'],budget=b)
    for mode in ['width_only','interval_only']:add('C_actions',['RC'],action_mode=mode)
    for sched in ['edf','uniform']:add('C_scheduler',['RC'],scheduler=sched)
    for p in [0,.25,.5,1]:add('C_reuse',['CR','RC'],reusable=p)
    add('D_archive',['RC'],control='no_archive');add('D_lgb',['RC'],head='lightgbm')
    for seed in [11,19]:add('D_seed',['CR','RC'],seed=seed)
    assert len(jobs)==830
    for k,j in enumerate(jobs):j['job_id']=k
    return jobs

def select_fixed(cfg,bundle,missions,out,center='I'):
    # Same validation missions and actual communication objective, no truth oracle policy.
    candidates=[]
    for pair in [(a,b) for a in cfg['radio']['subarray_axes'] for b in cfg['radio']['subarray_axes']]:
        for interval in cfg['simulation']['intervals_s']:
            if cfg['simulation']['primary_N']*cfg['simulation']['observation_cost_s']/interval>cfg['simulation']['sensing_budget']:continue
            candidate=copy.copy(bundle);candidate['fixed']={'pair':pair,'interval':interval};values=[]
            for mission in range(missions):
                world=World(cfg,'network/validation',mission)
                method='F' if center=='I' else center
                values.append(simulate(cfg,world,candidate,method,action_mode='fixed')['net_mbps'])
            candidates.append({'pair':pair,'interval':interval,'validation_mbps':float(np.mean(values))})
    key='fixed' if center=='I' else f'fixed_{center}'
    bundle[key]=max(candidates,key=lambda x:x['validation_mbps']);dump_json(Path(out)/f'fixed_{center}_validation.json',candidates)
    with (Path(out)/'bundle.pkl').open('wb') as f:pickle.dump(bundle,f)
    return bundle[key]

def single_action_matrix():
    return [{'block':'single_actions','job_id':830+4*m+j,'mission':20000+m,'N':1,'budget':.05,'seed':7,'method':'RC','action_mode':mode} for m in range(20) for j,mode in enumerate(['fixed','interval_only','width_only','joint'])]

def run_job(cfg,job,bundles_root,out,chronos=None):
    from .training import load_bundle
    kind=job.get('head','tabpfn'); path=Path(bundles_root)/f'{kind}_seed{job["seed"]}'/'bundle.pkl';bundle=load_bundle(path)
    cfg=copy.deepcopy(cfg);cfg['simulation']['primary_N']=job['N'];cfg['simulation']['sensing_budget']=job['budget']
    if job['block']=='single_actions':
        if 'fixed_RC' not in bundle:raise ValueError('Run RC fixed-policy validation before single-action diagnostics')
        bundle['fixed']=bundle['fixed_RC']
        bundle['action_reference']=bundle['fixed_RC']
    world=World(cfg,'network/test',job['mission'],reusable=job.get('reusable'))
    run_out=Path(out)/f"job_{job['job_id']:04d}"
    result=simulate(cfg,world,bundle,job['method'],chronos,run_out,job.get('control'),job.get('scheduler','value'),job.get('action_mode','joint'))
    result.update({'block':job['block'],'job_id':job['job_id'],'experiment':job})
    dump_json(run_out/'metrics.json',result)
    return result
