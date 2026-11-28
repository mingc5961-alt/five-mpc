from chronos_route_heads import *
from prefix849 import collect
from paired_models import trajectory
from panel_types import Geometry,angle_deg
from routeclock.radio import array_gain
from geometry_panel import life
from live849 import start as mark,finish
from tube import EPS,scores
import gzip,io,csv
COUNTS={1,2,3,4,6,8,12,24,40};AGES=(.5,1.,2.,3.,5.,8.,10.)
def packets(split,k):
    r=collect(split,k)
    path=OUT/f'E1/prefix_index/{split}/{k}/prefixes.pkl'
    hist=[Memory() for _ in range(16)]
    for u,o,n,_,ck,c,width in pickle.loads(path.read_bytes()):
        mem=hist[u];mem.on_paid_fix(o);q=mem.query();mem.issue(c)
        assert n==len(mem.h) and arrhash(c)==ck
        yield u,o,n,mem,q,c,width,ck
def target(w,u,t,c):
    d=np.arange(501)*.02;y=trajectory(w,u,t+d);pred=interp(c,d)
    loss=10*np.log10(1024/array_gain(pf.angle(pred),pf.angle(y),(32,32)))
    bad=np.flatnonzero(loss>3.)
    return float(d[max(0,bad[0]-1)] if len(bad) else 10.),not len(bad),np.abs(pf.difference(pf.angle(pred),pf.angle(y)))
def inputs(job):
    split,k=job;key=f'lifeinputs_{k}';mark(key)
    dest=OUT/f'E1/calibration/inputs/{k}.pkl';receipt=dest.with_suffix('.json')
    pr=collect(split,k)
    if receipt.exists():
        r=get(receipt);assert r['input_sha']==pr['sha256'] and r['sha256']==sha_file(dest);finish(key,receipt);return
    w,wm=world(split,k);items=[]
    if split=='tube_calibration':
        from grudata import schedule
        valid=[i for i,(_,t) in enumerate(schedule(w,41000+k)) if t+10<=180]
        selected=int(np.random.default_rng(271127+k).choice(valid))
    for i,(u,o,n,mem,q,c,width,ck) in enumerate(packets(split,k)):
        if o.t+10>180:break
        if split=='tube_calibration' and i!=selected:continue
        err=target(w,u,o.t,c)[2] if split=='tube_calibration' else np.abs(pf.difference(pf.angle(c),pf.angle(trajectory(w,u,o.t+pf.GRID))))
        items.append((o,n,o.t-mem.h[-2].t if n>1 else 0.,q,width,err,mem.key))
    atom(dest,pickle.dumps(items,protocol=5));put(receipt,dict(world=k,physics=wm['physical_sha256'],split=split,input_sha=pr['sha256'],sha256=sha_file(dest),samples=len(items)));finish(key,receipt)
def fit(arm):
    from lightgbm import LGBMRegressor
    key='lifehead_'+arm;mark(key)
    dest=OUT/f'E1/calibration/{arm}.pkl';receipt=dest.with_suffix('.json');xs=[];ys=[]
    for k in range(1200,1216):
        for o,n,gap,q,width,y,_ in pickle.loads((OUT/f'E1/calibration/inputs/{k}.pkl').read_bytes()):
            xs.append(features(arm,width,o,n,gap,q).astype('f4'));ys.append(np.log(np.maximum(EPS,y)).ravel())
    tree=LGBMRegressor(objective='quantile',alpha=.90,n_estimators=110,num_leaves=12,min_child_samples=85,learning_rate=.055,random_state=7,verbosity=-1,n_jobs=1)
    tree.fit(np.concatenate(xs),np.concatenate(ys));booster=tree.booster_;ratios=[];anchors=[]
    for k in range(1300,1316):
        [(o,n,gap,q,width,y,prefix)]=pickle.loads((OUT/f'E1/calibration/inputs/{k}.pkl').read_bytes())
        predicted=raw(booster,features(arm,width,o,n,gap,q)[None])
        ratios.append(scores(y[None],predicted)[0]);anchors.append([k,prefix,o.t])
    factors=np.maximum.accumulate(np.max(ratios,axis=0));assert np.isfinite(factors).all() and min(factors)>0
    atom(dest,pickle.dumps(dict(tree=booster,final=factors,rank=16,anchors=anchors),protocol=5))
    put(receipt,dict(arm=arm,training_prefixes=len(xs),worlds=list(range(1200,1216)),anchors=anchors,rank=16,ratios=np.asarray(ratios).tolist(),factors=factors.tolist(),sha256=sha_file(dest)));finish(key,receipt)
def panel(job):
    phase,split,k=job;key=dict(smoke='s1',development='dev1',confirmation='c1',**{'final-smoke':'f1'})[phase]+'_'+str(k);mark(key)
    pr=collect(split,k);w,wm=world(split,k)
    folder=OUT/'E1/scores'/phase;dest=folder/(str(k)+'.csv.gz');receipt=folder/(str(k)+'.json')
    scales=(.75,) if phase in ('smoke','final-smoke') else (.45,.6,.75,1.)
    from chronos_recipients import run_id
    spec=sha(compact([pr['sha256'],phase,run_id(phase),sha_file(Path(__file__))]))
    if receipt.exists():
        r=get(receipt);assert r['spec']==spec and sha_file(dest)==r['sha256'];finish(key,receipt);return r
    geom=[Geometry(w.times,w.positions[u],w.velocities[u]) for u in range(16)]
    from grudata import schedule
    times=[[t for u,t in schedule(w,41000+k) if u==v] for v in range(16)]
    radio=pf.Radio(w.cfg);buf=io.StringIO();writer=csv.writer(buf);names=['world','uid','n','t','age','h','label','group','natural','prefix','center','error_deg','true_life','censored','time_support','route_support','permutable','scale','arm','effective','hold','U','O']
    writer.writerow(names);nrows=0;aligned=0;full=0;native_checks=0
    sig={a:hashlib.sha256() for a in ARMS};index=[]
    for u,o,n,mem,q,c,width,ck in packets(split,k):
        index.append([u,n,o.t,mem.key,ck,str(c.dtype)])
        if o.t+10>180:continue
        full+=1
        # All full origins are signed into the audit; published age/horizon panel uses the registered counts.
        if n not in COUNTS:continue
        actual,censored,_=target(w,u,o.t,c)
        profiles={}
        for arm in LEARN:
            eff=inherited(arm,q)
            if eff not in profiles:profiles[eff]=issued(arm,mem,width,q,phase=='smoke')[:2]
        for m in scales:
            holds={'C-native':admitted(radio,c,width,get_native(),m)}
            holds['GEO']=geo(c,mem.h,holds['C-native'])
            for arm in LEARN:
                eff=inherited(arm,q)
                holds[arm]=holds[eff] if eff in holds else admitted(radio,c,*profiles[eff],m)
                assert eff==arm or holds[arm]==holds[eff]
            if m==.75 and phase in ('smoke','final-smoke'):
                ref=life(w,u,o.t,c,width,get_native(),radio)
                assert np.allclose(ref,(actual,holds['C-native'],int(censored)),rtol=0,atol=2e-14)
                native_checks+=1
            for age in AGES:
                for h in (0,2):
                    if age+h>10:continue
                    t=o.t+age+h;label=geom[u].label(o.t,t,w.detour_windows[u])
                    if w.groups[u]=='no_reusable_path' and label=='first_pass':label='no_reusable_path'
                    natural=int(n==len(times[u]) or times[u][n]>o.t+age)
                    err=float(angle_deg(c[int(2*(age+h))],trajectory(w,u,t)))
                    common=[k,u,n,o.t,age,h,label,w.groups[u],natural,mem.key,ck,err,actual,int(censored),*q[3:],m]
                    for arm in ARMS:
                        a=max(holds[arm]-age,0.);l=max(actual-age,0.)
                        row=[*common,arm,inherited(arm,q),holds[arm],max(l-a,0.),max(a-l,0.)]
                        writer.writerow(row);sig[arm].update(compact(row[:18]));nrows+=1
                    aligned+=1
    atom(dest,gzip.compress(buf.getvalue().encode(),compresslevel=3,mtime=0))
    # Re-read persisted rows and compare independently by arm and inclusion order.
    hashes={a:hashlib.sha256() for a in ARMS};counts={a:0 for a in ARMS}
    with gzip.open(dest,'rt') as f:
        for row in csv.DictReader(f):
            a=row['arm'];hashes[a].update(compact([row[v] for v in names[:18]]));counts[a]+=1
    assert len({s.hexdigest() for s in hashes.values()})==len({s.hexdigest() for s in sig.values()})==1 and set(counts.values())=={aligned}
    idx=OUT/f'E1/prefix_index/{split}/{k}/keys.json.gz';atom(idx,gzip.compress(compact(index),mtime=0))
    r=dict(spec=spec,world=k,physics=wm['physical_sha256'],prefix_receipt=pr['sha256'],prefix_index_sha=sha_file(idx),issued=len(index),full_windows=full,rows=nrows,rows_by_arm=counts,row_digest_by_arm={a:h.hexdigest() for a,h in hashes.items()},mismatches=0,original_life_checks=native_checks,sha256=sha_file(dest))
    put(receipt,r);finish(key,receipt);return r
if __name__=='__main__':print(panel(('smoke','development',1400)))
