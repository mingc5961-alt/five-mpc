"""Verify complete receipts and aggregate paired worlds before computing uncertainty."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from server_paths import ROOT,RUN

OUT=RUN/'analysis'
WORLDS=tuple(range(4000,4024))
COLS=['G_Mbps','R_pct','unavailable_pct','pointing_pct','W_s','S_s','fixes','P5_Mbps','Jain']
PAIRS=[('P03','P_M'),('P03','P_Fstar'),('Q01','Q02'),('Q01','Q05'),('N01','N02'),
       ('N01','N03'),('N01','N04'),('N01','N06'),('P01','P03'),('N01','N05')]


def estimate(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('invalid denominator')
    rng=np.random.default_rng(271127)
    b=np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975])
    return dict(mean=a.mean(),lo=b[0],hi=b[1],positive=int((a>0).sum()),n=len(a))


def load(p):
    return json.loads(p.read_bytes())


def measures(r):
    s=r['result'];bits=np.array(s['bits'])/(180e6);e=np.array(r['events'])
    t=e[:,1];u=e[:,0].astype(int);n=len(t);ledger=s['ledger_s']
    assert np.array_equal(u[:16],np.arange(16)) and (np.diff(t)>0).all()
    assert s['probes_per_user']==9000 and abs(sum(ledger)-180)<1e-6
    assert abs(n*.0031-ledger[0])<1e-7 and ledger[0]<=3.00001
    assert len(bits)==16 and abs(bits.sum()-s['G_mbps'])<1e-6
    return np.array([s['G_mbps'],100*s['R'],100*s['unavailable_fraction'],
      100*(s['R']-s['unavailable_fraction']),s['narrow_good_data_s'],ledger[0],n,
      np.quantile(bits,.05),bits.sum()**2/(len(bits)*(bits@bits))])


def closed(require_chronos):
    manifest=load(RUN/'linux_freeze.json');pol=dict(manifest['policies']);quota=manifest['anchor']['K']
    if require_chronos:pol['N05']=load(RUN/'chronos_freeze.json')
    per=[];shared={};count=0
    for k in WORLDS:
        physics=load(RUN/'worlds'/f'rebuild_holdout_{k}.json')['physical_sha256']
        for row,p in pol.items():
            weights={float(m):v for m,v in p['weights'].items()}
            assert abs(sum(weights.values())-1)<1e-8
            a={}
            for m in sorted(set(weights)|{.75}):
                if row=='N05':
                    r=load(RUN/'native'/f'rebuild_holdout_{k}_C_{m:.2f}.json')
                    assert r['model_source']=='pinned_Chronos_seed7'
                else:
                    matches=list((RUN/'receipts').glob(f'rebuild_holdout.{k}.{row}.B3.m{m:.2f}.*.json'))
                    assert len(matches)==1
                    r=load(matches[0]);assert r['row']==row
                assert r['world']==physics and r['k']==k
                a[m]=measures(r);count+=1
                if row[0] in 'PQ':assert len(r['events'])==quota
                if row in ('P03','P_M','P_Fstar'):
                    digest=r['event_sha256']
                    assert shared.setdefault(k,digest)==digest
            for mode,values in [('common075',a[.75]),('frozen',sum(p*a[m] for m,p in weights.items()))]:
                per.append(dict(world=k,row=row,mode=mode,**dict(zip(COLS,values))))
    per=pd.DataFrame(per);per.to_csv(OUT/'closed_worlds.csv',index=False)
    dist=[];contr=[]
    for (row,mode),d in per.groupby(['row','mode']):
        for key in COLS:
            dist.append(dict(row=row,mode=mode,metric=key,**estimate(d.sort_values('world')[key])))
    for l,r in PAIRS:
        if l not in pol or r not in pol:continue
        for mode in ('common075','frozen'):
            dat=per[per['mode']==mode].pivot(index='world',columns='row',values=COLS)
            for key in COLS:
                contr.append(dict(left=l,right=r,mode=mode,metric=key,
                                       **estimate(dat[key,l]-dat[key,r])))
    pd.DataFrame(dist).to_csv(OUT/'closed_arms.csv',index=False)
    pd.DataFrame(contr).to_csv(OUT/'closed_contrasts.csv',index=False)
    return dict(verified_holdout_endpoints=count,worlds=len(WORLDS),supplemental_rows=len(pol),
                identical_oldest_event_hashes_worlds=len(shared),k=quota)


def panel():
    df=[]
    for k in WORLDS:
        p=pd.read_csv(RUN/'panel'/f'{k}.csv.gz');q=pd.read_csv(RUN/'independent_panel'/f'{k}.csv.gz')
        common=['world','u','n','t','age','h','label','natural']
        if 'label_sampled5' in p and 'label_sampled5' in q:common.append('label_sampled5')
        assert (p[common]==q[common]).all().all()
        df.append(pd.concat([p,q.drop(columns=common)],axis=1))
    data=pd.concat(df,ignore_index=True)
    keyed=data.groupby(['label','n','age','h','natural'])
    keyed.agg(queries=('world','size'),worlds=('world','nunique')).to_csv(OUT/'panel_coverage.csv')
    panel_metrics=[c for c in data if c.startswith('e_')]
    for name in ('A','M','Fstar'):
        delta=data['adm_'+name]-data['real_'+name]
        data['excess_'+name]=delta.clip(lower=0);data['unused_'+name]=(-delta).clip(lower=0)
        data['expired_'+name]=data['adm_'+name].eq(0).astype(float)
        panel_metrics.extend(['excess_'+name,'unused_'+name,'expired_'+name])
    for a,b in [('M','A'),('Fstar','A'),('IMM','RAFT'),('no_age','with_age'),('Chronos','RAFT')]:
        name='d_'+a+'_minus_'+b;data[name]=data['e_'+a]-data['e_'+b];panel_metrics.append(name)
    summaries=[];world_rows=[]
    primary=data[(data.h==0)&data.n.isin([6,8,12,24,40])&data.age.isin([2,3,5])]
    for label in ('same_direction_revisit','first_pass','detour','rejoin','hover','all'):
        group=primary if label=='all' else primary[primary.label==label]
        for exposure in ('imposed','natural'):
            sample=group if exposure=='imposed' else group[group.natural==1]
            if not len(sample):continue
            cells=sample.groupby(['world','n','age'])[panel_metrics]
            means=cells.mean().groupby(level=0).mean()
            tails=sample.groupby(['world','n','age'])[[c for c in data if c.startswith('e_')]].quantile(.9).groupby(level=0).mean()
            means=means.join(tails.add_prefix('cell_p90_'))
            for name in means:
                summaries.append(dict(label=label,exposure=exposure,metric=name,queries=len(sample),
                       users=sample[['world','u']].drop_duplicates().shape[0],**estimate(means[name])))
            for k,v in means.iterrows():world_rows.append(dict(world=k,label=label,exposure=exposure,**v.to_dict()))
    interaction=[]
    for exposure in ('imposed','natural'):
        sub=primary if exposure=='imposed' else primary[primary.natural==1]
        for metric in ('d_M_minus_A','d_Fstar_minus_A','d_IMM_minus_RAFT','d_no_age_minus_with_age'):
            shared=sub[sub.label.isin(['first_pass','same_direction_revisit'])].groupby(
                ['world','n','age','label'])[metric].mean().unstack('label').dropna()
            paired=(shared.same_direction_revisit-shared.first_pass).groupby(level=0).mean()
            interaction.append(dict(metric=metric,exposure=exposure,cells=len(shared),**estimate(paired)))
    pd.DataFrame(interaction).to_csv(OUT/'revisit_minus_first.csv',index=False)
    counts=primary.groupby(['world','u']).label.agg(lambda s:s.eq('same_direction_revisit').sum())
    counts.rename('primary_revisit_queries').to_csv(OUT/'revisit_per_user.csv')
    pd.DataFrame(summaries).to_csv(OUT/'panel_summary.csv',index=False)
    pd.DataFrame(world_rows).to_csv(OUT/'panel_worlds.csv',index=False)
    return dict(paired_queries=len(data),worlds=len(WORLDS),distinct_prefixes=len(data[['world','u','n']].drop_duplicates()),
                primary_revisit_queries=int((primary.label=='same_direction_revisit').sum()))


def model_index():
    paths=[*sorted((RUN/'models').glob('*')),*sorted((ROOT/'reference/models').glob('*/*'))]
    found=[]
    for p in paths:
        if not p.is_file():continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
        found.append(dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=h.hexdigest()))
    pd.DataFrame(found).to_csv(OUT/'model_hashes.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-chronos',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    status=closed(args.include_chronos)
    if len(list((RUN/'independent_panel').glob('*.csv.gz')))==len(WORLDS):status.update(panel=panel())
    model_index()
    (OUT/'audit.json').write_text(json.dumps(status,indent=2))
    print(status)
