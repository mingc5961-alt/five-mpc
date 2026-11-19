"""Join predeclared matched cells, audit initialization and quantify service/history effects."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from 汇总 import ci, audit, read, save, ROOT, TAB

campaign=(ROOT/'固定历史补充实验').resolve()
measures=['net_mbps','worst10_mbps','sensing_pct','hold_fail_pct','narrow_pct','coverage_pct',
          'observations_service','age_0-1_pct','age_3+_pct','over_budget_frames','unavailable_s','data_s','rate_in_data_mbps']
keys=['cost_ms','budget_pct','scheduler','method']
def column(frame,metric):
    return frame.set_index('task')[metric].sort_index()

def boundary():
    records=[]
    for path in sorted((campaign/'boundary').glob('*.json')):
        for row in read(path):
            errors=row.pop('mse_deg2')
            records.extend(dict(**row,method=m,mse=v) for m,v in errors.items())
    detail=pd.DataFrame(records)
    assert len(detail)==30*16*6*3*3
    save(detail,'服务起点预测逐机')
    g=detail.groupby(['world','cost_ms','budget_pct','horizon_after_start','method'],as_index=False).agg(
        mse=('mse','mean'),forecast_age_s=('forecast_age_s','mean'),observations=('observations','mean'))
    g['rmse']=np.sqrt(g.mse)
    out=[]
    for (dt,m),part in g.groupby(['horizon_after_start','method']):
        p=part.pivot(index='world',columns=['cost_ms','budget_pct'],values='rmse')
        for c,b in p.columns:
            out.append(dict(horizon_s=dt,method=m,cost_ms=c,budget_pct=b,**ci(p[3.1,5.],p[c,b])))
    save(pd.DataFrame(out),'起点预测密集减原历史')
    h=detail.query('cost_ms==3.1 and budget_pct==5').groupby('world').history_sha256
    assert (h.nunique()==1).all() and h.first().nunique()==30
    return h.first().to_dict()

def collect():
    signatures=boundary()
    plan=read(campaign/'manifest.json')
    old=pd.read_csv(TAB/'逐场指标1500.csv')
    old['fixed_action']=False
    old['history_protocol']=old['warm'].map({'fixed':'统一密集','lifecycle':'全生命周期'})
    old['history_sha256']=None
    new=[]; users=[]; coverage=[]; gates=[]; proofs=[]
    path_links=[]
    for r in plan['new']:
        path=Path(r['out'])
        assert (path/'completed.json').is_file()
        audit(r,path)
        s=read(path/'warm_history.json')
        assert s['sha256']==signatures[r['world']] and s['counts']==[180]*8+[179]*8 and s['world']==r['world']
        a=read(path/'arguments.json')
        assert a['warm']=='fixed' and a['use_calibration']=='deadline' and a['split']=='test'
        m=read(path/'metrics.json')
        oldrow=old[old.metrics==str(Path(r['source'])/'metrics.json')].iloc[0]
        proofs.append(dict(out=str(path),reference=r['source'],world=r['world'],method=r['method'],policy=r['policy'],
                       history_sha256=s['sha256'],counts=','.join(map(str,s['counts']))))
        if r['policy']=='baseline_check':
            original=read(Path(r['source'])/'metrics.json')
            ignore={'mean_update_cpu_s','max_update_cpu_s'}
            assert all(v==original[k] for k,v in m.items() if k not in ignore) and set(m)==set(original)
            continue
        rid='固定历史/'+path.relative_to(campaign).as_posix()
        raw={k:v for k,v in m.items() if not isinstance(v,(list,dict))}
        d=dict(**{k:r[k] for k in ['world','split','model','calibration','warm','budget_pct','cost_ms']},**raw,
              id=rid,batch='固定历史',metrics=str(path/'metrics.json'),fixed_action=r['policy']=='fixed_action',
              history_protocol='统一密集',history_sha256=s['sha256'],reference_id=oldrow.id)
        cov=m['selected_prefix_coverage']
        count=sum(v['selected'] for v in cov.values())
        d.update(eligible_data_s=m['data_s']-m['unavailable_s'],sensing_pct=100*m['sensing_fraction'],
                 narrow_pct=100*m['narrow_share'],hold_fail_pct=100*m['hold_failures']/m['hold_segments'],
                 data_share=m['data_s']/180,rate_in_data_mbps=m['net_mbps']*180/m['data_s'],
                 coverage_pct=100*sum(v['covered'] for v in cov.values())/count if count else np.nan)
        for age,sec in m['age_data_s'].items():d['age_'+age+'_pct']=100*sec/d['eligible_data_s']
        new.append(d)
        users.extend(dict(id=rid,uid=i,**u,net_mbps=m['uav_mbps'][i]) for i,u in enumerate(m['group_resources']))
        coverage.extend(dict(id=rid,delta_s=float(k),**v) for k,v in cov.items())
        if r['method']=='RC':
            with np.load(path/'gate_weights.npz') as z:
                gates.extend(dict(id=rid,horizon_s=h,n_packages=len(z['weights']),mean_route_weight=v)
                             for h,v in zip(z['h'],z['weights'].mean(axis=0)))
        if not d['fixed_action']:
            path_links.append((rid,oldrow.id))
    save(pd.DataFrame(proofs),'固定历史逐场指纹')
    for rows,name in ((users,'逐机资源'),(coverage,'选中期限覆盖计数'),(gates,'逐场门控权重均值')):
        save(pd.concat([pd.read_csv(TAB/(name+'.csv')),pd.DataFrame(rows)],ignore_index=True),name+'合并')
    new=pd.DataFrame(new)
    base=old.query("split=='test' and cost_ms==3.1 and budget_pct==5").copy()
    assert len(base)==len(plan['reused'])==140
    refs={str(Path(r['out'])/'metrics.json') for r in plan['reused']}
    assert refs==set(base.metrics)
    base['reference_id']=base.id
    base['history_protocol']='统一密集';base['history_sha256']=base.world.map(signatures)
    dense=pd.concat([base,new[~new.fixed_action]],ignore_index=True)
    life=old[old.split=='test'].copy()
    assert len(dense)==len(life)==660 and len(new)==580
    assert not dense.duplicated(keys+['task']).any() and not life.duplicated(keys+['task']).any()
    assert set(dense.reference_id)==set(life.id)
    allruns=pd.concat([old,new],ignore_index=True)
    assert len(allruns)==allruns.id.nunique()==2080
    save(allruns,'合并逐场指标2080')
    return dense,life,new[new.fixed_action]

def analyze(dense,life,aux):
    contrasts=[]; histories=[]; effects=[]; common=[]; interactions=[]; anchor=[]
    for key,g in dense.groupby(keys):
        orig=life
        for k,v in zip(keys,key):orig=orig[orig[k]==v]
        assert set(g.task)==set(orig.task)
        for met in measures:
            a,b=column(g,met),column(orig,met)
            if a.isna().all():continue
            assert not a.isna().any() and not b.isna().any()
            histories.append(dict(zip(keys,key),metric=met,**ci(a,b)))
    for (c,b),group in dense[dense.scheduler=='value'].groupby(['cost_ms','budget_pct']):
        for met in measures[:6]:
            p=group.pivot(index='task',columns='method',values=met)
            for m,refs in [('RC',['I','P','F']),('C',['I','P','RC','F']),('CR_raw',['I','P','RC','F','C'])]:
                for r in refs:
                    if {m,r}.issubset(p):
                        q=p[[m,r]].dropna()
                        if not q.empty:contrasts.append(dict(cost_ms=c,budget_pct=b,metric=met,arm=m,ref=r,**ci(q[m],q[r])))
    for met in measures:
        filt=lambda d:d[(d.scheduler=='value') & (d.method.isin(['I','RC']))]
        dp=filt(dense).pivot(index='task',columns=['cost_ms','budget_pct','method'],values=met)
        lp=filt(life).pivot(index='task',columns=['cost_ms','budget_pct','method'],values=met)
        for c,b in dp.columns.droplevel(2).unique():
            gain=lambda p,k,h:p[k,h,'RC']-p[k,h,'I']
            def val(p,k,h,arm):return gain(p,k,h) if arm=='RC-I' else p[k,h,arm]
            for arm in ['I','RC','RC-I']:
                for label,array in [('密集减原',val(dp,c,b,arm)-val(lp,c,b,arm)),('服务相对基点',val(dp,c,b,arm)-val(dp,3.1,5.,arm))]:
                    effects.append(dict(cost_ms=c,budget_pct=b,arm=arm,metric=met,contrast=label,**ci(array,0*array)))
                if c==12.4:
                    for protocol,pv in [('密集',dp),('全生命周期',lp)]:
                        interactions.append(dict(arm=arm,metric=met,protocol=protocol,**ci(val(pv,12.4,5.,arm),val(pv,9.3,5.,arm))))
                    x=val(dp,12.4,5.,arm)-val(dp,9.3,5.,arm);y=val(lp,12.4,5.,arm)-val(lp,9.3,5.,arm)
                    interactions.append(dict(arm=arm,metric=met,protocol='密集减全生命周期的变化',**ci(x,y)))
            if met=='net_mbps' and c in (6.2,9.3):
                budget=2.5 if c==6.2 else dp.columns.get_level_values(1).min()
                anchor.append(dict(cost_ms=c,match_budget_pct=budget,**ci(gain(dp,c,5.),gain(dp,3.1,budget))))
    strict=dense[(dense.method=='F') & (dense.budget_pct<2)]
    fixed=pd.concat([aux,strict])
    assert len(fixed)==90
    for met in measures:
        p=fixed.pivot(index='task',columns='budget_pct',values=met)
        for b in p.columns:
            if p[b].notna().all():common.append(dict(budget_pct=b,metric=met,**ci(p[b],p[5.])))
    sched=[]
    for met in measures:
        p=dense[(dense.method=='RC') & (dense.budget_pct<2)].pivot(index='task',columns='scheduler',values=met)
        for other in ['edf','uniform']:sched.append(dict(metric=met,ref=other,**ci(p['value'],p[other])))
    for rows,name in [(histories,'固定历史与原协议配对'),(contrasts,'固定历史方法配对'),(effects,'历史及服务效应'),
            (interactions,'高成本两档交互'),(anchor,'固定历史等间隔对照'),(common,'同动作F预算对照'),(sched,'固定历史调度对照')]:
        save(pd.DataFrame(rows),name)
    means=dense.groupby(keys,as_index=False).agg(n=('task','size'),**{m:(m,'mean') for m in measures})
    save(means,'固定历史工况全指标')

if __name__=='__main__':
    assert read(campaign/'status.json')['stage']=='simulation_complete'
    analyze(*collect())
