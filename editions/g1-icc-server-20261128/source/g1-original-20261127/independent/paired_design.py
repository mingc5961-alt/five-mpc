"""Freeze all row policies and sample size using development worlds before opening confirmation scores."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv,json,math,multiprocessing,os,pickle,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from history import ROOT,ASSETS,PACKAGE,RUNTIME,put
from paired_grid import BASE,SCALES,load_checked
from physical_inputs import fingerprint,generate,sha_file
from slots import compute_slot
from policy_env import initialize
from deterministic_matrix import budgets,key as deterministic_key
from native_matrix import job_key
from learned_matrix import ROWS as LEARNED,series
from risk_matrix import evaluate
from predictors import Model

DESIGN=BASE/'design'
STRESS=ROOT.parent/'g1-server-handoff/runs/rebuild_v1/stress_strict'
VIEW={'structured_0':'structured0','structured_9_reference':'reference9','structured_16':'structured16','structured_9_detour':'detour9'}
FIXED={'N01':('Aorig','native'),'N02':('F0orig','native'),'N03':('Aorig','pm'),'N04':('Aorig','3s'),'N05':('Chronos','native'),
       'N06':('Aorig','free'),'P01':('Acal','fixed_due'),'P03':('Acal','oldest'),'Q01':('Acal','due'),'Q02':('M','due'),
       'Q05':('Fstar','due'),'Q09':('Aorig','due')}
def capacity():
    with (ROOT.parent/'g1-server-handoff/SERVER_CAPACITY.csv').open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def check_frozen_inputs(design):
    for filename,expected in design['simulation_and_model_hashes'].items():
        if sha_file(Path(filename))!=expected:
            raise ValueError(f'Frozen simulation or model changed: {filename}')
    for filename,expected in design['checked_inputs'].items():
        if sha_file(BASE/filename)!=expected:
            raise ValueError(f'Frozen development receipt changed: {filename}')
@lru_cache
def config(scene):
    bq=budgets()
    return dict(zip(['main','K700','K950','B4.5','B9'],bq)).get(scene,bq[0])
def seeds(row):
    return (17,29,41) if row in LEARNED else (None,)
def base_path(scene,k,row,s,b,q,m):
    if row=='N01' and b==3:return BASE/'receipts'/f'N01_development_{k}_{m:.2f}.json'
    if row in LEARNED:return BASE/'learned_matrix/receipts'/f'{row}_{k}_s{s}_B{b:g}_K{q}_m{m:.2f}.json'
    if row in ('Q06','Q10'):return BASE/'risk_matrix/receipts'/f'{row}_{k}_m{m:.2f}.json'
    if row.startswith('N') or row=='Q09':return BASE/'native_matrix/receipts'/(job_key((k,row,b,m))+'.json')
    return BASE/'deterministic_matrix/receipts'/(deterministic_key((k,row,b,q,m))+'.json')
def outputs(folder,k,row,seed,b,q,m):
    tag=f'{row}_{k}_s{seed}_B{b:g}_K{q}_m{m:.2f}'
    return folder/'receipts'/(tag+'.json')
def group_job(job):
    split,scene,k,row,seed,scales,folder=job;b,q=config(scene);folder=__import__('pathlib').Path(folder)
    for name in ('receipts','active'):(folder/name).mkdir(parents=True,exist_ok=True)
    if scene in VIEW:
        name=VIEW[scene];meta=json.loads((STRESS/f'{k}_{name}.json').read_bytes())
        with (STRESS/f'{k}_{name}.pkl').open('rb') as f:w=pickle.load(f)
        wh,parts=fingerprint(w)
        if wh!=meta['physical_sha256'] or w.mission!=k or w.groups!=meta['groups'] or w.namespace!=meta['namespace']:
            raise ValueError('stress world mismatch')
    else:w,meta=load_checked(split,k);wh=meta['physical_sha256']
    w.cfg['simulation']['sensing_budget']=b/180
    with compute_slot():
        if row in LEARNED:series(w,wh,split,k,row,seed,b,q,scales,folder);return
        import paid_full as pf
        from closed import simulate_controls
        for m in scales:
            dest=outputs(folder,k,row,seed,b,q,m)
            if dest.exists():
                r=json.loads(dest.read_bytes())
                if (r['world'],r['scale'],r['row'])!=(wh,m,row):raise ValueError(dest)
                continue
            mark=folder/'active'/dest.name;put(mark,dict(pid=os.getpid(),started=time.time()))
            try:
                period=json.loads((BASE/'summary.json').read_bytes())['PM_ms'] if row=='N03' else None
                if row in ('Q06','Q10'):
                    coverage=json.loads((BASE/'risk_matrix/threshold.json').read_bytes())['selected']
                    es,r,forced=evaluate(w,row,b,q,m,coverage)
                else:
                    kind,clock=FIXED[row];predictor=Model(w,kind)
                    es,r,forced=simulate_controls(w,clock,predictor.predict,predictor.factors,pf,b,m,
                        q if row[0] in 'PQ' else None,period)
                if row[0] in 'PQ' and (len(es)!=q or abs(r['ledger_s'][0]-q*.0031)>1e-7):raise ValueError('cost')
                if row=='N04' and len(es)!=960:raise ValueError('3-s clock')
                put(dest,dict(protocol='g1_linux_paired_v2',split=split,scenario=scene,row=row,seed=seed,world_id=k,world=wh,
                              budget_s=b,quota=q if row[0] in 'PQ' else None,scale=m,period_ms=period,events=es,result=r,forced=forced))
            finally:mark.unlink(missing_ok=True)
def run_groups(jobs):
    ctx=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(16,mp_context=ctx) as heavy,ProcessPoolExecutor(30,mp_context=ctx) as light:
        work=[(heavy if j[3] in ('N05','Q08') else light).submit(group_job,j) for j in jobs]
        for f in as_completed(work):f.result()
def policy(points):
    opts=[]
    for m,(g,r) in points.items():
        if r<=7:opts.append((g,-r,1,{m:1.}))
        for n,(v,s) in points.items():
            if n<=m or (r-7)*(s-7)>=0:continue
            p=(7-s)/(r-s);opts.append((p*g+(1-p)*v,-7.,0,{m:p,n:1-p}))
    if opts:
        pick=max(opts,key=lambda o:o[:3]);return dict(weights=pick[3],feasible=True,points=points)
    m=min(points,key=lambda m:(points[m][1],-points[m][0]));return dict(weights={m:1.},feasible=False,points=points)
def data(scene,row,scales,branch):
    b,q=config(scene);points={};world_scores={}
    for m in scales:
        rr=[]
        for k in range(1400,1408):
            scores=[]
            for s in seeds(row):
                src=base_path(scene,k,row,s,b,q,m) if m in SCALES else outputs(branch/scene,k,row,s,b,q,m)
                r=json.loads(src.read_bytes())
                if r['world']!=json.loads((BASE/'summary.json').read_bytes())['physical_sha256'][str(k)]:raise ValueError(src)
                scores.append([r['result']['G_mbps'],100*r['result']['R']])
            rr.append(np.mean(scores,axis=0))
        world_scores[m]=np.asarray(rr);points[m]=np.mean(rr,axis=0).tolist()
    pol=policy(points)
    world_mix=sum(p*world_scores[m] for m,p in pol['weights'].items())
    return pol,world_mix
def main():
    initialize()
    DESIGN.mkdir(parents=True,exist_ok=True)
    families=defaultdict(list);branch=BASE/'extensions'
    for row in capacity():
        if row['world_first']=='1400':families[(row['scenario'],row['row'][0])].append(row['row'])
    policies={};paired={};extend=[]
    for (scene,prefix),rows in families.items():
        should=any(not data(scene,r,SCALES,branch)[0]['feasible'] for r in rows)
        if should:extend.append([scene,prefix])
    put(DESIGN/'extension_plan.json',dict(families=extend,additional_multipliers=[.30,1.25],confirmation_read=False))
    tasks=[('development',scene,k,r,s,(.30,1.25),str(branch/scene)) for scene,prefix in extend
           for r in families[(scene,prefix)] for s in seeds(r) for k in range(1400,1408)]
    if tasks:run_groups(tasks)
    for (scene,prefix),rows in families.items():
        grid=(*SCALES,.30,1.25) if [scene,prefix] in extend else SCALES
        for r in rows:
            pp,worlds=data(scene,r,grid,branch)
            policies[f'{scene}/{r}']=pp;paired[f'{scene}/{r}']=worlds
    req={}
    for p1,p2 in [('Q03','Q04'),('Q01','Q03')]:
        d=paired['main/'+p1]-paired['main/'+p2];sd=d.std(axis=0,ddof=1)
        req[p1+'-'+p2]=dict(sd_G=float(sd[0]),sd_R_pp=float(sd[1]),required=[math.ceil((2.802*x/e)**2) for x,e in zip(sd,(20,.5))],
            sensitivity_sd_plus_25pct=[math.ceil((2.802*1.25*x/e)**2) for x,e in zip(sd,(20,.5))])
    needed=max(i for r in req.values() for i in r['required'])
    nw=next((n for n in (24,32,48) if n>=needed),48)
    if nw>24:
        with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(generate,[('confirmation',k) for k in range(2024,2000+nw)]))
    stress_index={f'{k}/{name}':json.loads((STRESS/f'{k}_{name}.json').read_bytes())['physical_sha256'] for k in range(3000,3012) for name in VIEW.values()}
    sources=[ROOT/'independent'/name for name in ('physical_inputs.py','paired_grid.py','paired_models.py','predictors.py','policy_env.py','learning.py','learned_matrix.py','risk_matrix.py','paired_design.py','confirmation.py','geometry_panel.py','panel_types.py','native_matrix.py','deterministic_matrix.py')]
    sources.extend([ROOT.parent/'g1-server-handoff/science/closed.py',ROOT.parent/'g1-server-handoff/science/aoi_policy.py'])
    weights=json.loads((BASE/'learning/summary.json').read_bytes())['formal']
    sources.extend(Path(path) for path in weights.values())
    sources.extend((BASE/'aligned_models').glob('*.pkl'))
    from experiment_env import check_models
    models=check_models(PACKAGE/'reference',ASSETS,with_chronos=True)
    sources.extend(Path(p) for p in models.values())
    sources.append(ASSETS/'chronos_snapshot/model.safetensors')
    sources.extend(RUNTIME/name for name in ('grurun.py','grudata.py','grutrain.py','paid_full.py'))
    sources.extend(p for p in (BASE/'auxiliary/models').iterdir() if p.is_file())
    sources.extend([BASE/'risk_matrix/threshold.json',BASE/'aligned_models/static.json',ROOT/'independent/geometry_statistics.py',ROOT.parent/'g1-server-handoff/SERVER_CAPACITY.csv'])
    frozen=dict(protocol='g1_linux_paired_v2',core_worlds=nw,default_worlds=24,estimated_world_requirement=needed,precision=req,
        main_quota=config('main')[1],PM_ms=json.loads((BASE/'summary.json').read_bytes())['PM_ms'],
        development_policies=policies,extension_endpoints=len(tasks)*2,confirmation_touched=False,
        supplemental_stress_inputs=stress_index,preregistered_statistics='world-paired bootstrap 20000 draws seed 271127; primary G lower >0, R upper <0.5pp; equal cost',
        simulation_and_model_hashes={str(p):sha_file(p) for p in sources},
        checked_inputs={name:sha_file(BASE/name) for name in ['summary.json','learning/summary.json','auxiliary/summary.json','aligned_models/summary.json','native_matrix/summary.json','deterministic_matrix/summary.json','learned_matrix/summary.json','risk_matrix/summary.json']})
    put(DESIGN/'frozen.json',frozen)
if __name__=='__main__':main()
