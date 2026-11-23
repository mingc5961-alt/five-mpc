import sys,json,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from protocol_local import log
A='新RC-专用管';B='新RC-公共管';C='Chronos-原生管';D='Chronos-公共管';E='Chronos-专用管'
def summarize():
    rows=[]
    for folder,arms in [(OUT.parent/'cells',{'v2-refit':A,'Chronos-native':C,'Chronos-2':E}),(OUT/'cells',{B:B,D:D})]:
        for f in sorted(folder.glob('*.json')):
            a=json.loads(f.read_text());method=arms.get(a['arm'])
            if not method:continue
            r=a['result'];b=a['budget']
            assert r['probes_per_user']==9000 and len(r['bits'])==16 and abs(sum(r['ledger_s'])-180)<1e-6
            assert abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-7 and r['ledger_s'][0]<={'T':3,'M':4.5,'L':9}[b]+.003101
            rows.append([a['world'],b,method,r['G_mbps'],100*r['R'],r['ledger_s'][0],r['hold_mean_s'],r['initial_center_sha256']])
    d=pd.DataFrame(rows,columns=['世界','预算','方法','吞吐Mbps','风险%','感知秒','保持秒','中心指纹'])
    assert len(d)==150 and not d.duplicated(['世界','预算','方法']).any() and set(d['世界'])==set(range(50,60))
    fp=d.pivot(index=['世界','预算'],columns='方法',values='中心指纹')
    assert (fp[A]==fp[B]).all() and (fp[C]==fp[D]).all() and (fp[C]==fp[E]).all()
    q=d.drop(columns=['世界','中心指纹']).groupby(['方法','预算']).mean()
    pairs=[];idx=np.random.default_rng(2381).integers(10,size=(5000,10))
    for b in 'TML':
        for x,y in [(A,C),(A,D),(B,D),(B,C),(B,A),(D,E),(D,C)]:
            for k in ('吞吐Mbps','风险%','感知秒','保持秒'):
                p=d[d['预算']==b].pivot(index='世界',columns='方法',values=k)
                z=(p[x]-p[y]).to_numpy()
                pairs.append([b,x,y,k,z.mean(),*np.quantile(z[idx].mean(1),[.025,.975])])
    pairs=pd.DataFrame(pairs,columns=['预算','方法','对照','指标','配对均差','95%下限','95%上限'])
    note=pd.DataFrame([['样本','主确认世界 50–59 全部纳入，逐世界配对；每世界 16 用户、预热与服务各 180 秒。'],['预算','T/M/L 的感知上限为 3/4.5/9 秒，MCS 余量 0 dB，价值调度，最早到期刷新，10 ms TDMA。'],['公共管','各自训练 LightGBM，使用相同的公共特征定义，包括观测、间隔及预测中心几何量。240 条路线训练，独立 400 条路线校准。'],['原生管','Chronos 本身的分位预测构造宽度，沿用其冻结的原生校准系数。'],['专用管','各自的 LightGBM 纳入模型内部诊断，并分别校准。'],['风险','公共评分点上不可用或有效波束损耗超过 3 dB 的比例；表中百分数，配对均差单位为百分点。'],['公平性','每个世界、预算内，同一中心的不同管在共同预热末端中心指纹相同。后续闭环观测由各自刷新时刻决定。']],columns=['项目','说明'])
    dst=OUT/'非专用管配对结果.xlsx'
    with pd.ExcelWriter(dst,engine='openpyxl') as w:
        for name,v in [('均值',q.reset_index()),('配对差与区间',pairs),('逐世界结果',d),('口径',note)]:
            v.to_excel(w,sheet_name=name,index=False);s=w.sheets[name];s.freeze_panes='B2';s.auto_filter.ref=s.dimensions
            for col in s.columns:s.column_dimensions[col[0].column_letter].width=min(95,max(16,max(len(str(c.value)) for c in col[:22])+4))
    print(q.round(3).to_string(),flush=True)
    log('非专用管补测 60/60；与主确认 90 场原生/专用对照逐世界合并。30 个世界×预算组合均已核对同中心指纹。')
    for b in 'TML':
        v=lambda k,m:q.loc[(k,b),m]
        log(f'{b} 档（逐世界等权），新 RC 专用 / RC 公共 / Chronos 原生 / Chronos 公共管的净吞吐 Mbps 和风险 % 依次为 '+ '，'.join(f"{v(k,'吞吐Mbps'):.2f}/{v(k,'风险%'):.3f}" for k in (A,B,C,D))+'。')
if __name__=='__main__':summarize()
