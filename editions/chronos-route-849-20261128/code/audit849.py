from chronos_recipients import *
from prefix849 import collect
from grudata import schedule
import resource
def validate(phase):
    w,wm=world('development',1400);ch=verified_cal()
    p=OUT/'E1/scores'/phase/'1400.json';life=get(p)
    assert life['mismatches']==0 and len(set(life['row_digest_by_arm'].values()))==1 and sha_file(p.with_suffix('.csv.gz'))==life['sha256']
    models={};observed={};prefixes={};checks=0;rr=[]
    for rule,seed in MATRIX:
        path=OUT/f'E2/receipts/{phase}/1400_{rule}_s{seed}_m0.75.json';r=get(path);file=OUT/f'E2/decision_logs/{phase}/{path.stem}.json.gz'
        assert sha_file(file)==r['log_sha'] and r['event_sha']==sha(compact(r['events']))
        log=json.loads(gzip.decompress(file.read_bytes()));seen=[EMPTY]*16
        assert len(r['events'])==len(log['issued'])==len(log['decisions'])==849 and r['physics']==wm['physical_sha256']
        for i,((u,a,b),(uid,pre,csha,osha),d) in enumerate(zip(r['events'],log['issued'],log['decisions'],strict=True)):
            assert uid==u==d['uid'] and i==d['i'] and [a,b]==cal[i] and (i>=16 or u==i)
            assert d['pre_history_sha']==seen
            o=w.observe(u,a/1e6+COST)
            assert sha(obsfull(o))==osha and digest(seen[u],o)==pre and round(o.t*1e6)==b
            assert (u,b) not in observed or observed[u,b]==osha
            assert (u,pre) not in prefixes or prefixes[u,pre]==csha
            observed[u,b]=osha;prefixes[u,pre]=csha;seen[u]=pre;checks+=1
        result=r['result'];l=result['ledger_s'];assert abs(sum(l)-180)<1e-6 and abs(l[0]-2.6319)<1e-7 and result['probes_per_user']==9000 and abs(sum(result['bits'])/180e6-result['G_mbps'])<1e-6
        rr.append(dict(arm=rule,seed=seed,starts_ends_checked=849,paid_s=l[0],ledger_s=l,total_s=sum(l),prefix_and_observation_errors=0,event_sha=r['event_sha'],seconds=r['elapsed_s'],cache_hits=r['cache_hits'],cache_misses=r['cache_misses'],old_entry_regression='exact events and full result' if rule in ('MAXAGE','EDF-native') else None))
        models[r['service_model_sha']]=1
    assert len(models)==1
    raw=pickle.loads((OUT/'E1/prefix_index/development/1400/prefixes.pkl').read_bytes());keys=json.loads(gzip.decompress((OUT/'E1/prefix_index/development/1400/keys.json.gz').read_bytes()))
    h=[EMPTY]*16;ev=list(schedule(w,41000+1400))
    for (u,o,n,_,cks,c,width),entry,(uid,t) in zip(raw,keys,ev,strict=True):
        assert (u,o.t)==(uid,t) and obsfull(w.observe(u,t))==obsfull(o);h[u]=digest(h[u],o);assert entry==[u,n,o.t,h[u],arrhash(c),str(c.dtype)]
    o=w.observe(0,cal[0][1]/1e6);mem=Memory();mem.on_paid_fix(o)
    try:mem.on_paid_fix(o)
    except ValueError:pass
    else:raise AssertionError('duplicate or backdated receipt accepted')
    forged=copy.copy(o);forged.radial_velocity+=1
    assert digest(EMPTY,forged)!=digest(EMPTY,o) and digest(EMPTY,w.observe(1,o.t))!=digest(EMPTY,o)
    rec=dict(status='PASS',phase=phase,initial_integration_heads=(phase=='smoke'),world=1400,physics=wm['physical_sha256'],calendar_sha256=ch,E1=life,E2=rr,verified_E2_rows=checks,distinct_paid_observations=len(observed),distinct_paid_prefixes=len(prefixes),unique_purchase_streams=len(set(r['event_sha'] for r in rr)),service_sha=list(models)[0],archive_regression=get(OUT/'audits/legacy.json'),negative_checks=['reject duplicate and nonmonotone paid time','changed raw observation changes complete prefix','wrong user changes complete prefix'],peak_rss_this_audit_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    put(OUT/f'audits/{phase}.json',rec);index=get(OUT/'audits.json') if (OUT/'audits.json').exists() else {};index[phase]=rec;put(OUT/'audits.json',index)
    return rec

def audit_confirm_world(k):
    return audit_grid_world(('confirmation',k))
def audit_grid_world(job):
    split,k=job
    from seal849 import seal
    seal((split,k))
    from policy_env import plan_events
    w,wm=world(split,k)
    if split=='confirmation':
        d=get(OUT/'design/frozen.json');required=[j for j in d['base_endpoints']+d['extra_endpoints'] if j[0]==k]
    elif split=='development':
        required=[[k,a,seed,m] for a,seed in MATRIX for m in (.45,.60,.75,1.)]
    else:raise ValueError(split)
    out=OUT/f'audits/{split}/{k}.json';names=[];seen={};centers={};issued=0;dqn={};rankings=[]
    e1=get(OUT/f'E1/scores/{split}/{k}.json')
    assert e1['mismatches']==0 and e1['physics']==wm['physical_sha256']
    for _,a,s,m in required:
        name=f'{k}_{a}_s{s}_m{m:.2f}';base=OUT/f'E2/receipts/{split}/{name}.json';p=OUT/f'E2/decision_logs/{split}/{name}.json.gz'
        r=get(base)
        assert r['physics']==wm['physical_sha256'] and sha_file(p)==r['log_sha'] and r['event_sha']==sha(compact(r['events']))
        rec=json.loads(gzip.decompress(p.read_bytes()));h=[EMPTY]*16;events=r['events']
        assert rec['service_model_sha']==r['service_model_sha']==service_id() and len(events)==849 and [v[1:] for v in events]==cal
        if a=='AoI-DQN-fixed':
            if s not in dqn:dqn[s]=plan_events(w,grid,'fixed',policy('fixed',s)[0])[0]
            assert [list(v) for v in dqn[s]]==events
        for i,((u,a0,b),(uid,prefix,csha,osha),dec) in enumerate(zip(events,rec['issued'],rec['decisions'],strict=True)):
            assert (u==uid==dec['uid'] and dec['i']==i and (i>=16 or i==u) and dec['pre_history_sha']==h)
            observations=seen.get((u,b))
            if observations is None:
                o=w.observe(u,a0/1e6+COST);observations=obsfull(o);seen[u,b]=observations
            assert sha(observations)==osha
            h[u]=sha(bytes.fromhex(h[u])+observations);assert h[u]==prefix
            prior=centers.setdefault((u,prefix),csha);assert prior==csha
            if a=='VALUE-route' and m==.75 and i>=16:
                for alternate,chosen in dec['same_state'].items():
                    gap=abs(dec['age'][u]-dec['age'][chosen])
                    rankings.append([k,i,u,alternate,chosen,gap,gap<=.186+1e-10,dec['support'][u][1],dec['support'][chosen][1]])
        issued+=len(events);names.append([name,sha_file(base),r['log_sha']])
    same=(OUT/f'E2/same_state_rankings/{k}.json.gz' if split=='confirmation' else OUT/f'E2/same_state_rankings/development/{k}.json.gz')
    assert len(rankings)==833*3
    atom(same,gzip.compress(compact(rankings),mtime=0))
    put(out,dict(status='PASS',rankings_sha=sha_file(same),world=k,physics=wm['physical_sha256'],endpoints=len(names),unique_prefixes=len(centers),endpoint_prefixes_checked=issued,observations=len(seen),E1_mismatches=0,E2_prefix_or_calendar_mismatches=0,ledger_verified_by_each_endpoint=True,receipts=names))
    return k
