"""Audit the frozen task grid, then export world- and flight-paired estimates."""
from pathlib import Path
from itertools import product
import argparse
import json
import warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
MODELS=('Chronos-2','VECTOR','RAFT')
METRICS=['net_mbps','risk_pct','unavailable_pct','geom_pct','geom_given_action_pct']
STAGES=('baselines','development','confirmation','amovfly')

def expected_keys():
    def grid(stage,units,models,seeds,tubes):
        return {(stage,str(u),b,m,s,t) for u,b,m,s,t in product(units,'TML',models,seeds,tubes)}
    keys=grid('baselines',range(40),('I','F','RC'),[0],['default'])
    wins=json.loads((ROOT/'work/flight_windows.json').read_text())
    ids=[r['flight_id'] for r in wins]
    keys|=grid('amovfly',ids,['I','RC'],[7],['default'])
    for m in MODELS:
        candidates=json.loads((ROOT/'tubes'/m/'7/offline.json').read_text())['candidates']
        keys|=grid('development',range(10),[m],[7],candidates)
        sel=ROOT/'tubes'/m/'selection.json'
        if sel.exists():
            tube=json.loads(sel.read_text())['tube']
            keys|=grid('confirmation',range(40),[m],[7,11,19],[tube])
            keys|=grid('amovfly',ids,[m],[7],[tube])
    return keys

def audit(full):
    rows=[];keys=set();warm={};centers={}
    flights={w['flight_id']:w for w in json.loads((ROOT/'work/flight_windows.json').read_text())}
    for stage in STAGES:
        for p in sorted((ROOT/'results'/stage).rglob('complete.json')):
            record=json.loads(p.read_text());res=record['result']
            if stage=='baselines':
                c=record['frozen_config'];m=c['model'];u=str(c['world']);b=c['condition'];s=0;t='default'
                assert c['namespace']=='confirm_0db_20261122/v1' and c['mcs_margin_db']==0
            elif stage=='amovfly':
                c=record['contract'];m=c['model'];u=c['flight_id'];b=c['budget'];s=c['seed'];t=c['tube']
                assert warm.setdefault(u,c['prewarm_sha256'])==c['prewarm_sha256']
                assert c['flight_sha256']==flights[u]['source_sha256']
            else:
                m=record['model'];u=str(record['world']);b=record['budget'];s=record['seed'];t=record['tube']
                assert record['center_sha256']==json.loads((ROOT/'training'/m/str(s)/'complete.json').read_text())['sha256']
                assert record['namespace']==('develop' if stage=='development' else 'confirm')+'_0db_20261122/v1'
                assert record['mcs_margin_db']==0
                h=res['initial_center_sha256']
                assert centers.setdefault((stage,m,u,b,s),h)==h
            key=stage,u,b,m,s,t
            assert key not in keys,key
            keys.add(key)
            n=res['probes_per_user']*len(res['bits'])
            assert n==9000*(1 if stage=='amovfly' else 16)
            un=round(n*res['unavailable_fraction']);risk=round(n*res['R']);bad=risk-un
            assert 0<=un<=risk<=n and abs(n*res['R']-risk)<1e-6 and abs(n*res['unavailable_fraction']-un)<1e-6
            assert abs(sum(res['bits'])/180/1e6-res['G_mbps'])<1e-7 and abs(sum(res['ledger_s'])-180)<1e-6
            budget={'T':.05/3,'M':.025,'L':.05}[b]/(16 if stage=='amovfly' else 1)
            assert res['ledger_s'][0]<=budget*180+.00310001
            rows.append(dict(stage=stage,unit=u,model=m,budget=b,seed=s,tube=t,N=n,U=un,B=bad,
                session=flights[u]['session'].split()[0] if stage=='amovfly' else '',net_mbps=res['G_mbps'],
                risk_pct=100*risk/n,unavailable_pct=100*un/n,geom_pct=100*bad/n,
                geom_given_action_pct=100*bad/(n-un) if un<n else np.nan))
    wanted=expected_keys()
    assert keys<=wanted,keys-wanted
    if full:
        assert len(wanted)==1620+15*len(flights) and keys==wanted,(len(keys),len(wanted),list(wanted-keys)[:5])
    return pd.DataFrame(rows)

def contrasts(data,phase):
    # Within-world seed means before paired whole-world resampling.
    group=data.groupby(['unit','budget','model'],as_index=False)[METRICS].mean()
    units=sorted(group.unit.unique())
    rng=np.random.default_rng(271828)
    ix=rng.integers(len(units),size=(5000,len(units)))
    clusters=data[['unit','session']].drop_duplicates().set_index('unit').reindex(units).session
    schemes={'paired_units':ix}
    if phase=='amovfly' and clusters.nunique()>1:
        names=sorted(clusters.unique()); samples=rng.integers(len(names),size=(5000,len(names)))
        # Repeated days contribute all their flights in every bootstrap draw.
        schemes['flight_date_clusters']=np.stack([(samples==i).sum(1) for i in range(len(names))],axis=1)[:,[names.index(x) for x in clusters]]
    records=[]
    for b in 'TML':
        v=group[group.budget==b]
        for m in MODELS:
            for base in (['I','RC'] if phase=='amovfly' else ['I','F','RC']):
                a=v[v.model==m].set_index('unit').reindex(units)[METRICS]
                z=v[v.model==base].set_index('unit').reindex(units)[METRICS]
                delta=(a-z).to_numpy()
                with warnings.catch_warnings(),np.errstate(divide='ignore',invalid='ignore'):
                    warnings.simplefilter('ignore',RuntimeWarning)
                    for key,draw in schemes.items():
                        if key=='paired_units':rep=np.nanmean(delta[draw],axis=1)
                        else:rep=(draw@np.nan_to_num(delta))/(draw@np.isfinite(delta).astype(float))
                        ci=np.nanquantile(rep,[.025,.975],axis=0)
                        for j,metric in enumerate(METRICS):
                            records.append(dict(phase=phase,budget=b,model=m,reference=base,metric=metric,
                                                bootstrap=key,n_units=int(np.isfinite(delta[:,j]).sum()),
                                                difference=np.nanmean(delta[:,j]),lower95=ci[0,j],upper95=ci[1,j]))
    return pd.DataFrame(records)

def main(partial):
    frame=audit(not partial)
    print(frame.groupby('stage').size().to_dict(),flush=True)
    if partial:return
    dest=ROOT/'results/summary';dest.mkdir(exist_ok=True)
    syn=frame[frame.stage.isin(['confirmation','baselines'])];flight=frame[frame.stage=='amovfly']
    tables={'all_cells':frame,'synthetic_seed_means':syn.groupby(['model','budget','seed'])[METRICS].mean().reset_index(),
            'amovfly_means':flight.groupby(['model','budget'])[METRICS].mean().reset_index(),
            'paired_effects':pd.concat([contrasts(syn,'synthetic'),contrasts(flight,'amovfly')])}
    for name,table in tables.items():
        tmp=dest/(name+'.tmp');table.to_csv(tmp,index=False);tmp.replace(dest/(name+'.csv'))
    out=dest/'complete.json';tmp=out.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(cells=len(frame),bootstrap_draws=5000,rng=271828,tables=list(tables))));tmp.replace(out)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');main(parser.parse_args().partial)
