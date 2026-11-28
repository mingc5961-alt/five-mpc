from chronos_recipients import *
from live849 import start as mark,finish
from policy_env import trace
from scipy.stats import spearmanr
from filelock import FileLock, Timeout
HORIZON=24
def clone(state,t0,stop):
    c=copy.copy(state);c.bucket=copy.copy(state.bucket);c.slots=[copy.copy(s) for s in state.slots]
    c.mem=[]
    for mem in state.mem:
        v=copy.copy(mem);v.pending=mem.pending.copy();v.records=mem.records.copy();v.h=mem.h.copy();c.mem.append(v)
    c.risk=state.risk.copy();c.sup=state.sup.copy();c.events=[];c.issued=[];c.audit=[]
    c.ledger=pf.DeferredLedger(c.w.cfg,c.w,c.radio,t0,stop-t0)
    full=state.ledger.probes;c.ledger.probes=full[(full>=t0)&(full<stop)]
    c.ledger.rr=state.ledger.rr
    return c
def cf(state,i,uid,check=False):
    start=cal[i][0]/1e6;tick=TICKS[i];last=tick+HORIZON
    end_idx=int(np.searchsorted(TICKS,last));stop=cal[end_idx][0]/1e6 if end_idx<len(cal) else 180.
    assert state.now==start and stop>=grid.start_us(last)/1e6-1e-12
    c=clone(state,start,stop)
    for j in range(i,end_idx):
        if j!=i:c.preservice(j)
        u=uid if j==i else int(np.argmin(c.deadline()))
        c.buy(j,u,heads=False)
    c.ledger.advance(c.now,stop,c.slots,False);c.ledger.result(COST,RHO)
    bits,fail=trace(c.ledger,grid);total=float(bits[tick:last].sum());fault=int(fail[tick:last].sum())
    receipt=dict(bits=total,fault=fault,events=c.events)
    if check:
        stream=state.events+c.events;hist=[Memory() for _ in state.mem];cap=[]
        def predict(u,o):
            hist[u].on_paid_fix(o);return state.cache(u,hist[u])
        def capture(*args):
            ledger=pf.DeferredLedger(*args);cap.append(ledger);return ledger
        # Preserve the full enclosing radio interval at the window end. An extra
        # archived-calendar event is used solely to reproduce that next boundary.
        if end_idx<len(cal):stream=stream+[[int(np.argmin(c.deadline())),*cal[end_idx]]]
        events,r,_=closed.simulate_controls(state.w,'stream',predict,state.cache.model.factors,pf,3.,state.m,849,event_stream=stream,ledger_factory=capture)
        slowbits,slowfail=trace(cap[0],grid)
        delta=abs(total-slowbits[tick:last].sum())
        assert [[u,round(t*1e6)] for u,t in events[i:i+len(c.events)]]==[[u,e] for u,_,e in c.events]
        assert delta<=1 and np.array_equal(fail[tick:last],slowfail[tick:last]) and np.max(np.abs(bits[tick:last]-slowbits[tick:last]))<=1
        receipt.update(slow_max_bin_delta=float(np.max(np.abs(bits[tick:last]-slowbits[tick:last]))),total_delta=float(delta),fail_differences=0)
    return receipt
def one_candidate(state,meta,root,u,blocking):
    k=meta['world'];i=meta['i'];key=f"{meta['group']}_{k}_{i}_{u}";path=root/f'{u}.json'
    try:
        with FileLock(str(path)+'.lock',timeout=-1 if blocking else 0):
            if path.is_file():
                r=get(path);assert (r['spec'],r['prefix'],r['candidate'])==(meta['spec'],meta['prefix'],u) and r['self_sha']==sha(compact(r['rollout']))
            else:
                mark(key)
                rollout=cf(state,i,u,check=meta['slow_check'] and u in (0,7))
                r=dict(spec=meta['spec'],prefix=meta['prefix'],physics=meta['physics'],world=k,ix=i,candidate=u,self_sha=sha(compact(rollout)),rollout=rollout)
                put(path,r)
            finish(key,path)
            return r
    except Timeout:
        if blocking:raise
        return None
def labelworld(job):
    from parallel849 import busy
    tag=busy('反事实主计算 '+str(job))
    try:return _labelworld(job)
    finally:tag.unlink(missing_ok=True)
def _labelworld(job):
    split,k,number=job;group={'train':'labels','train_validation':'val','development':'devcf'}[split]
    w,wm=world(split,k);cache=Cache(w,wm,split);state=Engine(w,cache,.75,False,'EDF-native')
    rng=np.random.default_rng(201127+k);eligible=np.flatnonzero((np.arange(len(TICKS))>=16)&(np.array(TICKS)+24<grid.opportunities))
    # Stratified over service time without choosing states by outcomes.
    chosen={int(rng.choice(v)) for v in np.array_split(eligible,number)}
    outs=[];spec=sha(compact([wm['physical_sha256'],modelid(),service_id(),sha_file(Path(__file__)),*[sha_file(OUT/f'E1/calibration/{a}.pkl') for a in LEARN]]))
    for i in range(max(chosen)+1):
        state.preservice(i);view=state.view(state.now)
        if i in chosen:
            root=OUT/f'E2/labels/{split}/{k}/{i}';x={side:state.values(state.now,side,view).tolist() for side in ('time','route')}
            pre=sha(compact([m.key for m in state.mem]));cs=[]
            snap=root/'snapshot.pkl';snapmeta=root/'snapshot.json'
            if not snapmeta.is_file():
                copy_state={name:getattr(state,name) for name in ('bucket','slots','mem','risk','sup','now','events','i')}
                atom(snap,pickle.dumps(copy_state,protocol=5))
                put(snapmeta,dict(spec=spec,prefix=pre,world=k,split=split,i=i,physics=wm['physical_sha256'],rr=state.ledger.rr,snapshot_sha=sha_file(snap),slow_check=k in (1000,1001,1400) and i==min(chosen),group=group))
            meta=get(snapmeta)
            assert meta['spec']==spec and meta['prefix']==pre
            for u in range(16):
                r=one_candidate(state,meta,root,u,blocking=True)
                cs.append(r['rollout']['bits'])
            item=dict(spec=spec,prefix=pre,world=k,appointment=i,features=x,bits=cs,labels=(np.asarray(cs)-np.mean(cs)).tolist(),edf=int(np.argmin(state.deadline())),support=state.sup.tolist(),ages=view[1].tolist())
            put(root/'state.json',item);outs.append(item)
        uid=i if i<16 else (int(rng.integers(16)) if rng.random()<.35 else int(np.argmin(state.deadline())))
        state.buy(i,uid)
    return len(outs)
def trainvalue(side):
    from lightgbm import LGBMRegressor
    key='value_'+side;mark(key)
    train=[get(p) for p in sorted((OUT/'E2/labels/train').glob('*/*/state.json'))]
    val=[get(p) for p in sorted((OUT/'E2/labels/train_validation').glob('*/*/state.json'))]
    assert len(train)==256 and len(val)==16 and len(set(r['prefix'] for r in train+val))==272
    x=np.concatenate([s['features'][side] for s in train]);y=np.concatenate([s['labels'] for s in train])/1e6
    net=LGBMRegressor(objective='regression',random_state=7,n_estimators=110,num_leaves=12,min_child_samples=30,learning_rate=.055,verbosity=-1,n_jobs=1).fit(x,y)
    dst=OUT/f'E2/labels/VALUE-{side}.txt';atom(dst,net.booster_.model_to_string().encode())
    ranks=[];regret=[]
    for r in val:
        p=net.predict(r['features'][side]);b=np.asarray(r['bits']);s=spearmanr(p,b).statistic
        if np.isfinite(s):ranks.append(s)
        regret.append(max(b)-b[np.argmax(p)])
    rec=dst.with_suffix('.json');put(rec,dict(side=side,train_worlds=32,states=256,labels=4096,validation_worlds=16,validation_rank_correlation=float(np.mean(ranks)) if ranks else None,validation_regret_bits=float(np.mean(regret)),sha256=sha_file(dst),train_sample_sha=sha(compact([[r['prefix'],r['labels']] for r in train]))));finish(key,rec)
