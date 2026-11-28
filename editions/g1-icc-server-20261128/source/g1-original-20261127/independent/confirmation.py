"""Execute the frozen, deduplicated confirmation and stress coordinates and audit paired claims."""
from collections import defaultdict
import csv,json,os,time
from pathlib import Path
import numpy as np
from history import put
from paired_grid import BASE
from paired_design import capacity,config,seeds,VIEW,outputs,run_groups,DESIGN,STRESS,check_frozen_inputs
from physical_inputs import sha_file
from policy_env import initialize
OUT=BASE/'confirmation'
PRIMARY={'Q01','Q02','Q03','Q04','Q05'}
METRICS=['G_mbps','R_pct','S_s','W_s','P5_mbps','Jain','unavailable_pct','misaligned_available_pct',
         'paid_fixes','p95_gap_s','max_gap_s','paid_last_30s']

def plan(design):
    work=defaultdict(set);overall=[]
    weights=design['development_policies']
    for spec in capacity():
        if int(spec['world_first'])<2000:continue
        scene=spec['scenario'];row=spec['row'];first=int(spec['world_first']);last=int(spec['world_last'])
        if scene=='main' and row in PRIMARY and spec['stage'] in ('core','common'):last=1999+design['core_worlds']
        common=spec['multiplier_rule']=='0.75'
        mix={.75:1.} if common else {float(m):p for m,p in weights[f'{scene}/{row}']['weights'].items() if p>0}
        for k in range(first,last+1):
            for s in seeds(row):work[(scene,k,row,s)].update(mix)
        overall.append(dict(scene=scene,row=row,stage=spec['stage'],first=first,last=last,weights=mix,seeds=list(seeds(row))))
    return work,overall

def metrics(r):
    v=r['result'];e=r['events'];s=v['ledger_s'][0];bits=np.asarray(v['bits'])
    assert len(bits)==16 and v['probes_per_user']==9000 and abs(sum(v['ledger_s'])-180)<1e-6
    assert abs(s-len(e)*.0031)<1e-7 and abs(bits.sum()/180e6-v['G_mbps'])<1e-6
    by_user=[[] for _ in range(16)]
    for uid,t in e:by_user[uid].append(t)
    gaps=np.concatenate([np.diff([0.,*events,180.]) for events in by_user])
    if np.min(gaps)<-1e-8 or v['unavailable_fraction']>v['R']+1e-10:raise ValueError('invalid gap or risk components')
    jain=bits.sum()**2/(len(bits)*np.dot(bits,bits)) if bits.any() else 0.
    return [v['G_mbps'],v['R']*100,s,v['narrow_good_data_s'],np.quantile(bits/180e6,.05),jain,
        100*v['unavailable_fraction'],100*(v['R']-v['unavailable_fraction']),len(e),
        np.quantile(gaps,.95),gaps.max(),sum(t>=150 for _,t in e)]
def summarize(work,groups,design):
    scores={};world_hash={};event_hash={};fixed_times={}
    for (scene,k,row,seed),scales in work.items():
        b,q=config(scene)
        for m in scales:
            r=json.loads(outputs(OUT/scene,k,row,seed,b,q,m).read_bytes())
            if r['world']!=world_hash.setdefault((k,scene if scene in VIEW else 'main_physics'),r['world']):raise ValueError('paired world mismatch')
            if row[0] in 'PQ' and (r['quota']!=q or len(r['events'])!=q):raise ValueError('quota mismatch')
            if row in ('Q03','Q04','Q07','Q08') and r['event_sha256']!=event_hash.setdefault((scene,k,seed),r['event_sha256']):raise ValueError('policy stream mismatch')
            if row.startswith('P'):
                times=tuple(round(t*1e6) for u,t in r['events'])
                if times!=fixed_times.setdefault((scene,k),times):raise ValueError('P success times mismatch')
            scores[(scene,k,row,seed,m)]=np.asarray(metrics(r))
    mixed={};outrows=[]
    cols=METRICS
    for item in groups:
        scene,row,stage=item['scene'],item['row'],item['stage']
        for k in range(item['first'],item['last']+1):
            v=np.mean([sum(p*scores[(scene,k,row,s,float(m))] for m,p in item['weights'].items()) for s in item['seeds']],axis=0)
            mixed[(scene,stage,row,k)]=v
            outrows.append(dict(scenario=scene,stage=stage,row=row,world=k,**dict(zip(cols,v))))
    with (OUT/'world_scores.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
    contrasts=[]
    for a,b in [('Q03','Q04'),('Q01','Q03'),('Q01','Q05'),('N01','N02'),('N01','N03'),('N01','N04'),('N01','N05'),
                ('N01','N06'),('N06','Q09'),('Q09','Q01'),('P01','P02'),('Q01','Q06')]:
        paired=[]
        for k in range(2000,2000+design['core_worlds']):
            def v(row):
                return next((mixed[('main',stage,row,k)] for stage in ('core','supp') if ('main',stage,row,k) in mixed),None)
            left,right=v(a),v(b)
            if left is not None and right is not None:paired.append(left-right)
        arr=np.asarray(paired);rng=np.random.default_rng(271127)
        if not len(arr):raise ValueError((a,b))
        idx=rng.integers(len(arr),size=(20000,len(arr)));ci=np.quantile(arr[idx].mean(axis=1),[.025,.975],axis=0)
        result=dict(contrast=f'{a}-{b}',worlds=len(arr),mean=dict(zip(cols,arr.mean(axis=0))),
                    lower=dict(zip(cols,ci[0])),upper=dict(zip(cols,ci[1])))
        if (a,b) in [('Q03','Q04'),('Q01','Q03')]:
            result['passed_both_primary_bounds']=bool(ci[0,0]>0 and ci[1,1]<.5 and np.max(abs(arr[:,2]))<1e-7)
        if (a,b)==('N01','N03'):
            result['sensing_time_equivalent_within_0_03s']=bool(ci[0,2]>-.03 and ci[1,2]<.03)
        contrasts.append(result)
    put(OUT/'contrasts.json',contrasts)
    # Each estimate resamples whole worlds; multiple seeds and operating points are already averaged within a world.
    def comparison(scene,stage,coefficients):
        rows=tuple(coefficients)
        keys=sorted({k for s,t,r,k in mixed if (s,t,r)==(scene,stage,rows[0])})
        selected=[k for k in keys if all((scene,stage,r,k) in mixed for r in rows)]
        if not selected:return None
        d=np.array([sum(w*mixed[(scene,stage,r,k)] for r,w in coefficients.items()) for k in selected])
        idx=np.random.default_rng(271127).integers(len(d),size=(20000,len(d)))
        ci=np.quantile(d[idx].mean(axis=1),[.025,.975],axis=0)
        return dict(world_ids=selected,weights=coefficients,positive_G_worlds=int((d[:,0]>0).sum()),
            mean=dict(zip(cols,d.mean(axis=0))),lower=dict(zip(cols,ci[0])),upper=dict(zip(cols,ci[1])))
    diagnostics=[]
    for scene,stage in sorted({(s,t) for s,t,r,k in mixed}):
        qs=[{'Q03':1,'Q04':-1},{'Q01':1,'Q03':-1},{'Q01':1,'Q02':-1},{'Q01':1,'Q05':-1},
            {'Q01':1,'Q02':-1,'Q03':-1,'Q04':1},{'P01':1,'P02':-1},{'P01':1,'P03':-1}]
        for coef in qs:
            item=comparison(scene,stage,coef)
            if item:diagnostics.append(dict(scenario=scene,stage=stage,**item))
    operating_points=[]
    for scene,stage,row in sorted({(s,t,r) for s,t,r,k in mixed}):
        item=comparison(scene,stage,{row:1})
        operating_points.append(dict(scenario=scene,stage=stage,row=row,
            upper_absolute_risk_within_7_pct=bool(item['upper']['R_pct']<=7),**item))
    put(OUT/'operating_points.json',operating_points)
    for name in ('Q07','Q08','Q10'):
        ref='Q02' if name=='Q10' else 'Q03'
        for k in range(2000,2012):
            if ('main','core',ref,k) in mixed:mixed['main','supp',ref,k]=mixed['main','core',ref,k]
        item=comparison('main','supp',{ref:1,name:-1})
        if item:diagnostics.append(dict(scenario='main',stage='supp',**item))
    put(OUT/'secondary_contrasts.json',diagnostics)
    return dict(verified_endpoints=len(scores),world_series=len(outrows),world_fingerprints=len(world_hash),
                paired_joint_streams=len(event_hash),paired_fixed_schedules=len(fixed_times),contrasts=len(contrasts),
                secondary_contrasts=len(diagnostics))
def main():
    initialize()
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=DESIGN/'frozen.json';design=json.loads(frozen.read_bytes());digest=sha_file(frozen)
    check_frozen_inputs(design)
    for name,expected in design['supplemental_stress_inputs'].items():
        k,view=name.split('/')
        if json.loads((STRESS/f'{k}_{view}.json').read_bytes())['physical_sha256']!=expected:
            raise ValueError(f'Frozen stress world receipt changed: {name}')
    work,groups=plan(design)
    jobs=[('stress' if scene in VIEW else 'confirmation',scene,k,row,s,tuple(sorted(ms)),str(OUT/scene))
          for (scene,k,row,s),ms in work.items()]
    n=sum(len(m) for m in work.values());start=time.time()
    manifest=OUT/'queue.json'
    plan_data=dict(design_sha256=digest,unique_endpoints=n,groups=groups,primary_worlds=design['core_worlds'])
    if manifest.exists() and json.loads(manifest.read_bytes())['design_sha256']!=digest:raise ValueError('frozen design changed')
    put(manifest,plan_data)
    put(OUT/'batch.json',dict(total=n,workers=46,pid=os.getpid(),start=start,phase='confirmation'))
    run_groups(jobs)
    verified=summarize(work,groups,design)
    if verified['verified_endpoints']!=n:raise ValueError('missing endpoints')
    put(OUT/'summary.json',dict(**verified,design_sha256=digest,elapsed_s=time.time()-start,core_worlds=design['core_worlds'],
                                cross_platform_reference_status='separate from locally paired results'))
    put(OUT/'batch.json',dict(total=n,workers=46,pid=os.getpid(),start=start,phase='done'))
if __name__=='__main__':main()
