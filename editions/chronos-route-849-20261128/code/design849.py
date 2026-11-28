from chronos_recipients import *
from chronos_life_panel import panel
from live849 import start as mark,finish
from parallel849 import capacity,shard
from physical_inputs import BANK
import pandas as pd
SCALES=(.45,.60,.75,1.)
@lru_cache(maxsize=1)
def frozen_physics():
    return {r['world']:(r['physical_sha256'],r['file_sha256'],r['namespace']) for r in get(OUT/'design/frozen.json')['worlds']}
def verify_world_freeze(meta):
    if (OUT/'design/frozen.json').is_file():
        assert frozen_physics()[meta['world']]==(meta['physical_sha256'],meta['file_sha256'],meta['namespace'])
def newworld(k):
    ns='chronos_route_849_v1/confirmation';d=OUT/'design/worlds';p=d/f'{k}.world';meta=p.with_suffix('.json')
    if meta.exists():
        r=get(meta);verify_world_freeze(r);assert sha_file(p)==r['file_sha256']
        w=pickle.loads(p.read_bytes());assert fingerprint(w)==(r['physical_sha256'],r['components']) and w.mission==k and w.namespace==ns
        return w,r
    assert 6000<=k<6096
    cfg=load_config();cfg['simulation'].update(warmup_s=0,service_s=180,sensing_budget=3/180)
    rng=np.random.default_rng(np.random.SeedSequence([20261127,k]));grp=[GROUPS[i] for i in rng.permutation(np.repeat(np.arange(6),[3,1,1,1,3,7]))]
    w=World(cfg,ns,k,groups=grp);digest,components=fingerprint(w);atom(p,pickle.dumps(w,protocol=5))
    r=dict(world=k,namespace=ns,physical_sha256=digest,components=components,file_sha256=sha_file(p),group_seed=[20261127,k],groups=grp);verify_world_freeze(r);put(meta,r);return w,r
def fingerprint_only(k):return newworld(k)[1]
def execute_endpoint(job):
    phase,split,k,r,s,m=job
    group={'development':'dev2','final-smoke':'f2','confirmation':'c2' if m==.75 else 'extra','smoke':'s2'}[phase]
    key=f'{group}_{k}_{r}_{s}_{m}'
    from filelock import FileLock
    locks=OUT/'jobs/locks';locks.mkdir(exist_ok=True)
    with FileLock(str(locks/(key+'.lock'))):
        from sharedcache849 import SharedPrefixCache
        expected=OUT/f'E2/receipts/{phase}/{k}_{r}_s{s}_m{m:.2f}.json'
        if not expected.exists():mark(key)
        stem=endpoint(job,cache_factory=SharedPrefixCache);finish(key,OUT/f'E2/receipts/{phase}/{stem}.json');return stem
def primary(data):
    return data[(data.label=='same_direction_revisit')&data.n.isin([6,8,12,24,40])&data.age.isin([2.,3.,5.])&(data.h==0)]
def life_summary(phase):
    ws=[]
    for p in sorted((OUT/'E1/scores'/phase).glob('*.csv.gz')):
        d=primary(pd.read_csv(p))
        if not d.empty:ws.append(d.groupby(['world','scale','arm','n','age'])[['U','O']].mean().groupby(['world','scale','arm']).mean().reset_index())
    return pd.concat(ws,ignore_index=True) if ws else pd.DataFrame(columns=['world','scale','arm','U','O'])
def receipts(phase):
    result=[]
    for path in sorted((OUT/'E2/receipts'/phase).glob('*.json')):
        r=get(path);z=r['result'];ev=r['events'];bits=np.asarray(z['bits']);u=np.array([v[0] for v in ev]);t=np.array([v[2] for v in ev])/1e6
        assert [v[1:] for v in ev]==cal and list(u[:16])==list(range(16)) and abs(sum(z['ledger_s'])-180)<1e-6 and abs(z['ledger_s'][0]-2.6319)<1e-7
        gaps=[float(np.diff(np.r_[0,t[u==uid],180]).max()) for uid in range(16)]
        result.append(dict(world=r['world'],arm=r['arm'],seed=r['seed'],scale=r['scale'],physics=r['physics'],G=z['G_mbps'],R=z['R']*100,unavailable=100*z['unavailable_fraction'],mispointed=100*(z['R']-z['unavailable_fraction']),P5=float(np.percentile(bits/180e6,5)),Jain=float(bits.sum()**2/(16*(bits@bits))),narrow_s=z['narrow_good_data_s'],purchases=len(ev),paid_s=z['ledger_s'][0],switch_s=z['ledger_s'][1],data_s=z['ledger_s'][2],unavailable_s=z['ledger_s'][3],max_gap_s=max(gaps),per_uid_counts=np.bincount(u,minlength=16).tolist(),per_uid_max_gaps=gaps,per_uid_bits=bits.tolist()))
    return pd.DataFrame(result)
def weights(points,target,reward,risk):
    # Match the target in expectation at world level, maximizing utility at that risk.
    src=sorted((float(i),float(row[reward]),float(row[risk])) for i,row in points.iterrows());cand=[]
    for m,g,r in src:
        if abs(r-target)<1e-10:cand.append((g,1,{m:1.}))
    for i,(m,g,r) in enumerate(src):
        for n,h,s in src[i+1:]:
            if min(r,s)<target<max(r,s):
                p=float((target-s)/(r-s));cand.append((p*g+(1-p)*h,0,{m:p,n:1-p}))
    if not cand:return None
    return max(cand,key=lambda c:(c[0],c[1]))[2]
def match(points,target,reward,risk):
    g=points.groupby('arm');lower=float(g[risk].min().max());upper=float(g[risk].max().min())
    target=min(max(target,lower),upper) if lower<=upper else None
    return dict(overlap=[lower,upper] if lower<=upper else None,target=target,ranges={a:[float(v[risk].min()),float(v[risk].max())] for a,v in g},weights=({a:weights(v.set_index('scale'),target,reward,risk) for a,v in g} if target is not None else {}))
def model_files():
    return sorted([*(OUT/'E1/calibration').glob('*.pkl'),*(OUT/'E2/labels').glob('VALUE-*.txt')])
def freeze():
    key='freeze_design';mark(key);dst=OUT/'design/frozen.json'
    if dst.is_file():
        d=get(dst);check(d);finish(key,dst);return d
    from gate849 import poolmap
    life=life_summary('development')
    e2=receipts('development')
    assert len(e2)==320 and len(list((OUT/'E1/scores/development').glob('*.csv.gz')))==8
    lifepts=life.groupby(['arm','scale']).mean(numeric_only=True).reset_index()
    if len(lifepts):
        a=lifepts[(lifepts.scale==.75)&lifepts.arm.isin(['C-cal','TIME'])]
        cap=float(a[a.arm=='TIME'].O.iloc[0]);strong1=a[a.O<=cap].sort_values(['U','O','arm']).arm.iloc[0]
        lifepts['minus_U']=-lifepts.U;match1=match(lifepts[lifepts.arm.isin(LEARN)],cap,'minus_U','O')
    else:cap=None;strong1='TIME';match1=dict(overlap=None,weights={})
    pts=e2.groupby(['arm','world','scale'])[['G','R']].mean().groupby(['arm','scale']).mean().reset_index()
    no=['MAXAGE','AoI-DQN-fixed','VAR','EDF-native','EDF-time','VALUE-time']
    def strong(arms):
        cs=pts[(pts.scale==.75)&pts.arm.isin(arms)];ok=cs[cs.R<=7]
        return str((ok.sort_values(['G','R','arm'],ascending=[False,True,True]) if len(ok) else cs.sort_values(['R','G','arm'],ascending=[True,False,True])).arm.iloc[0])
    match2=match(pts,7,'G','R')
    # Separately freeze each arm's best attainable point subject to R<=7, or its minimum risk.
    own={}
    for a,v in pts.groupby('arm'):
        clean={float(row.scale):[{'result':{'G_mbps':float(row.G),'R':float(row.R)/100}}] for row in v.itertuples()}
        own[a]=closed.freeze(clean)
    devdiff=e2[e2.scale==.75].groupby(['world','arm'])[['G','R']].mean().unstack().sort_index()
    dispersion={m:float((devdiff[m]['VALUE-route']-devdiff[m]['VALUE-time']).std()) for m in ('G','R')}
    elig=int(life.world.nunique());sd1=None
    if elig>=2:
        a=life[life.scale==.75].pivot(index='world',columns='arm',values='U');sd1=float((a.TIME-a.ROUTE).std())
    rule=get(OUT/'design/pre_fit_rules.json')['N_planning'];z=1.959963984540054+.8416212335729143
    target_g=rule['target_goodput_relative_to_VALUE_time']*float(devdiff['G']['VALUE-time'].mean())
    needs={'goodput':int(np.ceil((z*dispersion['G']/target_g)**2)),'risk':int(np.ceil((z*dispersion['R']/.5)**2))}
    if elig>=2:
        p=life[life.scale==.75].pivot(index='world',columns='arm',values='O')
        sd_o=float((p.ROUTE-p.TIME).std());occ=elig/8.
        needs.update(life_U=int(np.ceil((z*sd1/.10)**2/occ)),life_O=int(np.ceil((z*sd_o/.02)**2/occ)))
    else:needs.update(life_U=None,life_O=None)
    sufficient=[n for n in rule['world_count_grid'] if all(v is not None and v<=n for v in needs.values())]
    n=sufficient[0] if sufficient else max(rule['world_count_grid'])
    planning=dict(requirements=needs,selected=n,capacity_sufficient=bool(sufficient),target_goodput_mbps=target_g,primary_eligible_fraction=elig/8.,rules_sha=sha_file(OUT/'design/pre_fit_rules.json'))
    worlds=list(range(6000,6000+n))
    bank=get(BANK/'index.json')['worlds']
    existing={x['physical_sha256'] for x in bank};assert not set(worlds)&{x['world'] for x in bank}
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(20,mp_context=multiprocessing.get_context('fork')) as pool:metas=list(pool.map(fingerprint_only,worlds))
    assert len({r['physical_sha256'] for r in metas})==len(worlds) and not existing&{r['physical_sha256'] for r in metas}
    extra=[]
    for k in worlds:
        for a,s in MATRIX:
            scales=set(float(x) for x in own[a]['weights'])
            if a in match2['weights']:scales.update(float(x) for x in match2['weights'][a])
            extra.extend([[k,a,s,m] for m in sorted(scales-{.75})])
    from seal849 import seal
    calanchors=[get(OUT/f'E1/calibration/{a}.json')['anchors'] for a in LEARN]
    assert all(anchors==calanchors[0] for anchors in calanchors)
    assert len(calanchors[0])==16 and set(a[0] for a in calanchors[0])==set(range(1300,1316))
    assert get(OUT/'E2/labels/VALUE-time.json')['train_sample_sha']==get(OUT/'E2/labels/VALUE-route.json')['train_sample_sha']
    with ProcessPoolExecutor(12,mp_context=multiprocessing.get_context('fork')) as pool:
        list(pool.map(seal,[('tube_fit',k) for k in range(1200,1216)]+[('tube_calibration',k) for k in range(1300,1316)]+[('development',k) for k in range(1400,1408)]))
    manifest=get(OUT/'manifest.json');manifest['new_code']={p.name:sha_file(p) for p in sorted((OUT/'code').glob('*.py'))};manifest['code_sha256']=codehash()
    for p in model_files():manifest['files'][str(p)]=sha_file(p)
    put(OUT/'manifest.json',manifest)
    # All source files are additions to the pinned research implementation.
    chunks=[subprocess.run(['git','diff','--no-index','/dev/null',str(p)],capture_output=True).stdout for p in sorted((OUT/'code').glob('*.py'))]
    atom(OUT/'design/new_code.patch',b''.join(chunks))
    # Commit only source/configuration in this independent directory.
    run=lambda args: subprocess.check_output(['git',*args],cwd=OUT,text=True).strip()
    run(['init','-q']);run(['add','--','code/*.py','protocol.json','design/pre_fit_rules.json'])
    changes=subprocess.run(['git','diff','--cached','--quiet'],cwd=OUT)
    if changes.returncode==1:
        run(['-c','user.name=Chronos Experiment','-c','user.email=experiment@localhost','commit','-qm','Freeze Chronos-849 implementation and protocol'])
    elif changes.returncode!=0:raise RuntimeError('unable to verify staged sources')
    actual_commit=run(['rev-parse','HEAD'])
    d=dict(new_code_commit=actual_commit,protocol=P,code_hash=codehash(),manifest_sha=sha_file(OUT/'manifest.json'),models={str(p):sha_file(p) for p in model_files()},n=n,power_planning=planning,worlds=metas,splits={s:[r['physical_sha256'] for r in bank if r['split']==s] for s in ('train','train_validation','tube_fit','tube_calibration','development')},development_sha={str(p):sha_file(p) for p in sorted([*(OUT/'E2/receipts/development').glob('*.json'),*(OUT/'E1/scores/development').glob('*.json')])},strong_E1=str(strong1),strong_E2=strong(no),strong_AoI=strong(no[:2]),strong_nonroute_scheduling=strong(['EDF-native','EDF-time','VALUE-time']),life_over_cap=cap,E1_matched=match1,E2_matched=match2,E2_individual_target=own,base_endpoints=[[k,a,s,.75] for k in worlds for a,s in MATRIX],extra_endpoints=extra,common_scale=.75,bootstrap_seed=271127,bootstrap_replicates=20000,E1_primary='same_direction_revisit; n in 6,8,12,24,40; age in 2,3,5; h=0; equal eligible cells per world',effective_shuffle_min_worlds=8,effective_shuffle_min_prefixes=32,E1_delta_U_lower_gt=0,E1_delta_O_upper_lt=.02,E2_delta_G_lower_gt=0,E2_delta_R_upper_lt_pp=.5,absolute_R_upper_le_pct=7,dev_dispersion=dispersion,dev_E1_sd_U=sd1,dev_revisit_worlds=elig,N_rationale='Use the pre-fit bounded world-count grid and development paired dispersions, including revisit occupancy. No post-confirmation enlargement.',figure2_CF='two presampled stratified development states per world; all sixteen branches labeled under common EDF continuation; same-state VALUE-route minus VALUE-time conditional on unequal choices; split by route support',training_diagnostic_only='initial wiring smoke heads',inputs_disjoint=True)
    put(dst,d);finish(key,dst);return d
def check(d):
    assert codehash()==d['code_hash'] and all(sha_file(Path(p))==s for p,s in d['models'].items()) and sha_file(OUT/'manifest.json')==d['manifest_sha']
def development_freeze_confirm():
    from gate849 import poolmap
    from audit849 import validate
    (OUT/'logs/status.txt').write_text('开发集全臂与四倍率，并行选择工作点。\n')
    jobs=[('development','development',k,r,s,m) for m in SCALES for r,s in MATRIX for k in range(1400,1408)]
    mixed=[('panel',('development','development',k)) for k in range(1400,1408)]+[('endpoint',j) for j in jobs]
    poolmap(confirm_job,mixed,capacity())
    d=freeze();check(d)
    subprocess.run([sys.executable,'-Bu',str(OUT/'code/frozen_run.py')],cwd=OUT,check=True)
def frozen_execution():
    from gate849 import poolmap
    from audit849 import validate
    d=get(OUT/'design/frozen.json');check(d)
    history=[get(p) for p in BANK.glob('*/*.json') if p.stem.isdigit()]
    old_ids={r['world'] for r in history};new_ids={r['world'] for r in d['worlds']}
    old_sha={r['physical_sha256'] for r in history};new_sha={r['physical_sha256'] for r in d['worlds']}
    assert not old_ids&new_ids and not old_sha&new_sha and len(new_sha)==d['n']
    put(OUT/'design/split_audit.json',dict(all_historical_inputs=len(history),historical_confirmation=sum(r['split']=='confirmation' for r in history),new_worlds=d['n'],id_overlap=0,physics_overlap=0,new_namespace=d['worlds'][0]['namespace']))
    from audit849 import audit_grid_world
    for k in range(1400,1408):
        auditfile=OUT/f'audits/development/{k}.json'
        if not auditfile.is_file():audit_grid_world(('development',k))
        audit=get(auditfile);assert audit['status']=='PASS' and audit['endpoints']==40 and audit['E2_prefix_or_calendar_mismatches']==0
    pilot=get(OUT/'audits/smoke.json')
    med=float(np.median([r['seconds'] for r in pilot['E2']]))
    count=len(d['base_endpoints'])+len(d['extra_endpoints'])
    examples=[get(p)['elapsed_s'] for p in (OUT/'E2/receipts/development').glob('*.json')]
    put(OUT/'design/compute_budget.json',dict(frozen_worlds=d['n'],E2_endpoints=count,smoke_median_endpoint_wall_s=med,development_endpoint_quartiles_s=np.quantile(examples,[.25,.5,.75]).tolist(),physical_cpu_quota=48,primary_parallel_slots=47,E2_smoke_based_wall_hours=med*count/47/3600,formula='median 1400 native smoke endpoint time * frozen E2 endpoints / 47 native slots; excludes E1 and final rendering; cache sharing and CPU saturation change actual time'))
    poolmap(execute_endpoint,[('final-smoke','development',1400,r,s,.75) for r,s in MATRIX],10)
    panel(('final-smoke','development',1400));validate('final-smoke');check(d)
    (OUT/'logs/status.txt').write_text('最终模型单世界核验通过。新留出世界的 E1、E2 已放行。\n')
    from report849 import report
    ids=[w['world'] for w in d['worlds']]
    declared=set(tuple(j) for j in d['base_endpoints']+d['extra_endpoints'])
    tasks=[]
    for block in range(0,len(ids),8):
        worlds=ids[block:block+8]
        tasks.extend(('panel',('confirmation','confirmation',k)) for k in worlds)
        tasks.extend(('endpoint',('confirmation','confirmation',k,r,s,.75)) for r,s in MATRIX for k in worlds)
    for k,r,s,m in sorted(d['extra_endpoints'],key=lambda j:(j[3],j[1],j[2],j[0])):
        tasks.append(('endpoint',('confirmation','confirmation',k,r,s,m)))
    assert {tuple(j[2:]) for task,j in tasks if task=='endpoint'}==declared
    poolmap(confirm_job,tasks,capacity());check(d);report()
def confirm_job(task):
    with_step,job=task
    if with_step!='panel':return execute_endpoint(job)
    from filelock import FileLock
    locks=OUT/'jobs/locks';locks.mkdir(exist_ok=True)
    with FileLock(str(locks/('_'.join(str(v) for v in job)+'.lock'))):
        return panel(job)
