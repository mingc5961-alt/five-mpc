"""One row per completed physical world and arm; pair by the full world ID."""
import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routeclock.evaluation import paired_bootstrap


@lru_cache(None)
def model_settings(path):
    with path.open('rb') as stream:
        model=pickle.load(stream)
    gate=model.get('gate_mode','full')
    return 'G0' if gate=='full' else gate,model.get('head_mode','local'),model.get('calibration','global')


def summarize(root):
    rows=[]
    managed=(root/"manifest.json").exists()
    for path in sorted(root.rglob('metrics.json')):
        if managed and ('.work.' in path.parent.name or not (path.parent/'completed.json').exists()):continue
        args_path=path.parent/'arguments.json'
        if not args_path.exists(): continue
        args=json.loads(args_path.read_text())
        if args['stage']!='network' or args['smoke']:continue
        m=json.loads(path.read_text())
        cov=m.get('selected_prefix_coverage') or {}
        n=sum(x['selected'] for x in cov.values())
        model_path=Path(args['bundle'])
        if not model_path.is_absolute():
            model_path=Path(__file__).resolve().parents[1]/model_path
        gate,head,default_calibration=model_settings(model_path)
        rows.append(dict(run=path.parent.relative_to(root).as_posix(),world=m['task'],split=args['split'],
            method=m['method'],scheduler=m['scheduler'],bundle=str(args['bundle']),gate=gate,head=head,calibration=args.get('use_calibration') or default_calibration,
            warm=args['warm'],cost_ms=args['cost_ms'],budget_pct=100*m['sensing_budget'],
            **{f'age_{k}_data_pct':100*v/max(1e-12,m['data_s']-m['unavailable_s']) for k,v in m.get('age_data_s',{}).items()},
            net_mbps=m['net_mbps'],sensing_pct=100*m['sensing_fraction'],
            narrow_pct=100*m['narrow_share'],hold_fail_pct=100*m['hold_failures']/m['hold_segments'] if m['hold_segments'] else np.nan,
            worst10_mbps=m['worst10_mbps'],overdue_requests=m['overdue_requests'],unavailable_s=m['unavailable_s'],
            coverage_selected_n=n,coverage_pct=100*sum(x['covered'] for x in cov.values())/n if n else np.nan))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    args=p.parse_args()
    df=summarize(args.root)
    if df.empty:raise ValueError('No complete full-length runs')
    df.to_csv(args.root/'world_metrics.csv',index=False)
    paired=[]
    keys=['split','scheduler','bundle','calibration','warm','cost_ms','budget_pct']
    for key,g in df.groupby(keys,dropna=False):
        for metric in ['net_mbps','sensing_pct','hold_fail_pct','narrow_pct','worst10_mbps']:
            pivot=g.pivot(index='world',columns='method',values=metric)
            if 'RC' not in pivot:continue
            for ref in pivot.columns.difference(['RC']):
                pairs=pivot[['RC',ref]].dropna()
                if pairs.empty:continue
                delta=pairs.RC-pairs[ref]
                row=dict(zip(keys,key),metric=metric,reference=ref,n=len(pairs),mean_delta=delta.mean())
                if len(pairs)>1: row.update(paired_bootstrap(pairs.RC,pairs[ref]))
                paired.append(row)
    pd.DataFrame(paired).to_csv(args.root/'paired.csv',index=False)
    changes=[]
    for key,g in df[df.scheduler=='value'].groupby(['split','bundle','calibration'],dropna=False):
        pivot=g.pivot(index='world',columns=['warm','cost_ms','budget_pct','method'],values='net_mbps')
        points=g[['warm','cost_ms','budget_pct']].drop_duplicates().itertuples(index=False,name=None)
        for warm,cost,budget in points:
            if (cost,budget)==(3.1,5.):continue
            cols=[('lifecycle',3.1,5.,'RC'),('lifecycle',3.1,5.,'I'),(warm,cost,budget,'RC'),(warm,cost,budget,'I')]
            if not all(c in pivot for c in cols):continue
            rows=pivot[cols].dropna()
            if rows.empty:continue
            base=rows[cols[0]]-rows[cols[1]]
            changed=rows[cols[2]]-rows[cols[3]]
            record=dict(split=key[0],bundle=key[1],calibration=key[2],warm=warm,cost_ms=cost,budget_pct=budget,
                        n=len(rows),mean_change=(changed-base).mean())
            if len(rows)>1:record.update(paired_bootstrap(changed,base))
            changes.append(record)
    pd.DataFrame(changes).to_csv(args.root/'cost_interaction.csv',index=False)

    if (args.root/'manifest.json').exists():
        write_campaign_results(args.root,df)


def write_campaign_results(root,df):
    confirm=df[df.split=='test'].copy()
    if confirm.empty:return
    metrics=['net_mbps','sensing_pct','narrow_pct','hold_fail_pct','worst10_mbps','coverage_pct',
             'age_0-1_data_pct','age_1-2_data_pct','age_2-3_data_pct','age_3+_data_pct']
    groups=['cost_ms','budget_pct','scheduler','method']
    means=confirm.groupby(groups)[metrics].mean()
    means['n_worlds']=confirm.groupby(groups).world.nunique()
    means.reset_index().to_csv(root/'test_means.csv',index=False)
    chronos_pairs=[]
    for (cost,budget),g in confirm[confirm.scheduler=='value'].groupby(['cost_ms','budget_pct']):
        for metric in ('net_mbps','sensing_pct','hold_fail_pct','narrow_pct','worst10_mbps'):
            table=g.pivot(index='world',columns='method',values=metric)
            for method in ('C','CR_raw'):
                for reference in ('I','P','RC','F'):
                    if not {method,reference}.issubset(table.columns):continue
                    matched=table[[method,reference]].dropna()
                    if matched.empty:continue
                    left,right=matched[method],matched[reference]
                    chronos_pairs.append(dict(cost_ms=cost,budget_pct=budget,method=method,
                        reference=reference,metric=metric,n_worlds=len(matched),
                        method_mean=left.mean(),reference_mean=right.mean(),
                        **paired_bootstrap(left,right)))
    pd.DataFrame(chronos_pairs).to_csv(root/'chronos_paired.csv',index=False)
    primary=confirm[(confirm.scheduler=='value') & confirm.method.isin(['I','RC'])]
    by_world=primary.pivot(index='world',columns=['cost_ms','budget_pct','method'],values='net_mbps')
    matches=[]
    for cost,budget in [(6.2,2.5),(9.3,5/3)]:
        cols=[(cost,5.,'RC'),(cost,5.,'I'),(3.1,budget,'RC'),(3.1,budget,'I')]
        if not all(c in by_world for c in cols):continue
        x=by_world[cols].dropna()
        if x.empty:continue
        pressure=x[cols[0]]-x[cols[1]]
        low_budget=x[cols[2]]-x[cols[3]]
        matches.append(dict(cost_ms=cost,matched_budget_pct=budget,n_worlds=len(x),
                       pressure_rc_minus_i=pressure.mean(),budget_rc_minus_i=low_budget.mean(),
                       **paired_bootstrap(pressure,low_budget)))
    pd.DataFrame(matches).to_csv(root/'matched_spacing.csv',index=False)
    sched=confirm[(confirm.cost_ms==3.1)&np.isclose(confirm.budget_pct,5/3)&(confirm.method=='RC')]
    contrasts=[]
    for metric in metrics:
        table=sched.pivot(index='world',columns='scheduler',values=metric)
        for other in ('edf','uniform'):
            if not {'value',other}.issubset(table.columns):continue
            pairs=table[['value',other]].dropna()
            if len(pairs):contrasts.append(dict(metric=metric,reference=other,n_worlds=len(pairs),
                                                **paired_bootstrap(pairs['value'],pairs[other])))
    pd.DataFrame(contrasts).to_csv(root/'scheduler_paired.csv',index=False)
    if len(confirm)==600 and confirm.world.nunique()==30:
        lines=['# 30 个独立世界、600 场确认实验','',
               '| 成本 ms | 预算 % | 世界 | RC 吞吐 | I 吞吐 | 配对差 Mbps | 95% 区间 |',
               '|---:|---:|---:|---:|---:|---:|---|']
        for (cost,budget),g in primary.groupby(['cost_ms','budget_pct']):
            pivot=g.pivot(index='world',columns='method',values='net_mbps').dropna()
            b=paired_bootstrap(pivot.RC,pivot.I)
            assert len(pivot)==30
            lo,hi=b['ci95']
            lines.append(f'| {cost:g} | {budget:.4f} | 30 | {pivot.RC.mean():.3f} | {pivot.I.mean():.3f} | {b["difference"]:+.3f} | [{lo:.3f}, {hi:.3f}] |')
        lines+=['','按物理世界配对自助重采样 2000 次。吞吐、保持段失效率、最差两机吞吐和四个实际服务年龄段见 test_means.csv。',
                '等刷新间隔的成本与预算对照见 matched_spacing.csv。三类调度策略的配对差见 scheduler_paired.csv。Chronos 两个版本分别对 I/P/RC/F 的同世界配对差见 chronos_paired.csv。']
        (root/'确认结果.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
