"""Rebuild chapter-six data tables from audited per-world results. Uses numpy and pandas."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = (ROOT / '主实验归档').resolve()
MAIN = CODE / 'runs/paired600'
EXT = (ROOT / '追加实验归档').resolve()
TAB = ROOT / '数据表'
sys.path[:0] = [str(CODE / 'tools'), str(CODE)]
from ch6_run import audit
from routeclock.evaluation import paired_bootstrap


def read(path):
    return json.loads(path.read_text())


def save(df, name):
    df.to_csv(TAB / (name + '.csv'), index=False, encoding='utf-8-sig')


def ci(left, right):
    # Published summaries pivot on string task IDs, sorted lexicographically.
    left, right = left.align(right, join='inner')
    left, right = left.sort_index(), right.sort_index()
    b = paired_bootstrap(left, right)
    return dict(n=len(left), left=left.mean(), right=right.mean(), delta=b['difference'],
                lo=b['ci95'][0], hi=b['ci95'][1], wins=int((left>right).sum()))


def collect():
    entries, users, coverage, gates = [], [], [], []
    for directory, batch in ((MAIN,'原计划'), (EXT,'追加')):
        plan = read(directory / 'manifest.json')
        for stage, rows in plan.items():
            for r in rows:
                absolute = Path(r['out'])
                rel = Path(*absolute.parts[absolute.parts.index('paired600' if directory == MAIN else EXT.name)+1:])
                folder = directory / rel
                assert (folder / 'completed.json').is_file()
                audit(r, folder)
                if stage == 'fixed':
                    continue
                m = read(folder / 'metrics.json')
                args = read(folder / 'arguments.json')
                assert not args['smoke'] and args['stage']=='network'
                rid = f'{batch}/{rel}'
                record = {k:v for k,v in r.items() if k not in ('out','argv')}
                record.update({k:v for k,v in m.items() if not isinstance(v,(dict,list))})
                record.update(id=rid, batch=batch, metrics=str(folder / 'metrics.json'),
                              eligible_data_s=m['data_s']-m['unavailable_s'],
                              sensing_pct=100*m['sensing_fraction'], narrow_pct=100*m['narrow_share'],
                              hold_fail_pct=100*m['hold_failures']/m['hold_segments'],
                              data_share=m['data_s']/m['service_s'], rate_in_data_mbps=m['net_mbps']*m['service_s']/m['data_s'])
                counts=m['selected_prefix_coverage']
                selected=sum(v['selected'] for v in counts.values())
                record['coverage_pct']=100*sum(v['covered'] for v in counts.values())/selected if selected else np.nan
                for age, sec in m['age_data_s'].items():
                    record[f'age_{age}_pct']=100*sec/record['eligible_data_s']
                entries.append(record)
                for i,g in enumerate(m['group_resources']):
                    users.append(dict(id=rid, uid=i, **g, net_mbps=m['uav_mbps'][i]))
                for deadline, tally in counts.items():
                    coverage.append(dict(id=rid, delta_s=float(deadline), **tally))
                if r['method']=='RC':
                    with np.load(folder/'gate_weights.npz') as z:
                        ws=z['weights']
                        assert ws.shape==(len(z['origin']),len(z['h']))
                        for h,w in zip(z['h'],ws.mean(axis=0)):
                            gates.append(dict(id=rid,horizon_s=h,n_packages=len(ws),mean_route_weight=w))
    d=pd.DataFrame(entries)
    assert len(d)==d.id.nunique()==1500
    assert len(users)==len(d)*16
    for frame,name in ((d,'逐场指标1500'),(pd.DataFrame(users),'逐机资源'),(pd.DataFrame(coverage),'选中期限覆盖计数'),(pd.DataFrame(gates),'逐场门控权重均值')):
        save(frame,name)
    return d


def analyze(d):
    test=d[d.split=='test']
    cols=['net_mbps','hold_fail_pct','sensing_pct','narrow_pct','worst10_mbps','coverage_pct','observations_service',
          'overdue_requests','over_budget_frames','missing_action_frames','unavailable_s','data_share','rate_in_data_mbps',
          'age_0-1_pct','age_1-2_pct','age_2-3_pct','age_3+_pct']
    keys=['cost_ms','budget_pct','scheduler','method']
    def means(df,keys):
        return df.groupby(keys,as_index=False).agg(n=('task','size'),**{k:(k,'mean') for k in cols})
    save(means(test,keys),'测试工况全指标')
    save(means(d[d.split=='validation'],keys+['model','calibration','warm']),'验证工况全指标')
    contrasts, decompositions=[],[]
    for (cost,budget),g in test[test.scheduler=='value'].groupby(['cost_ms','budget_pct']):
        for metric in cols[:6]:
            p=g.pivot(index='task',columns='method',values=metric)
            for a,b in [('RC','I'),('RC','P'),('RC','F'),('C','I'),('C','RC'),('C','P'),('C','F'),
                        ('CR_raw','I'),('CR_raw','RC'),('CR_raw','P'),('CR_raw','F'),('CR_raw','C')]:
                if {a,b}.issubset(p):
                    z=p[[a,b]].dropna()
                    if len(z):
                        contrasts.append(dict(cost_ms=cost,budget_pct=budget,metric=metric,arm=a,ref=b,**ci(z[a],z[b])))
        def column(metric):
            return g.pivot(index='task',columns='method',values=metric)
        time,rate,net=map(column,['data_share','rate_in_data_mbps','net_mbps'])
        time_term=(time.RC-time.I)*(rate.RC+rate.I)/2
        rate_term=(rate.RC-rate.I)*(time.RC+time.I)/2
        assert np.allclose(time_term+rate_term,net.RC-net.I)
        for piece,x in [('数据时间项',time_term),('数据期有效速率项',rate_term)]:
            decompositions.append(dict(cost_ms=cost,budget_pct=budget,term=piece,**ci(x,x*0)))
    save(pd.DataFrame(contrasts),'测试方法配对')
    save(pd.DataFrame(decompositions),'吞吐差对称分解')
    v=d[(d.split=='validation')&(d.method=='RC')&(d.scheduler=='value')&(d.warm=='lifecycle')]
    ablations=[]
    for (cost,budget),g in v.groupby(['cost_ms','budget_pct']):
        for metric in cols[:6]:
            pivot=g.pivot(index='task',columns=['model','calibration'],values=metric)
            for a,b in [(('local','global'),('G0','global')),(('G0','deadline'),('G0','global')),
                      (('local','deadline'),('local','global')),(('local','deadline'),('G0','deadline')),
                      (('full','deadline'),('local','deadline'))]:
                if a in pivot and b in pivot:
                    ablations.append(dict(cost_ms=cost,budget_pct=budget,metric=metric,arm='/'.join(a),ref='/'.join(b),**ci(pivot[a],pivot[b])))
    save(pd.DataFrame(ablations),'验证机制配对')


if __name__=='__main__':
    analyze(collect())
