"""Summarize new paired mechanism diagnostics together with the frozen closed-loop ledgers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl.styles import Font,PatternFill
from openpyxl.utils import get_column_letter

out=Path(__file__).resolve().parent
root=Path('/data/chen/routeclock_runs/confirm_0db')
def read_shards(stem):
    return pd.concat([pd.read_csv(out/f'{stem}_{i}.csv',dtype={'unit':str}) for i in range(4)],ignore_index=True)
def interval(x):
    x=np.asarray(x);draw=np.random.default_rng(271828).integers(len(x),size=(5000,len(x)))
    return np.quantile(x[draw].mean(1),[.025,.975])
def comparison(frame,keys,metrics):
    records=[]
    for labels,d in frame.groupby(keys):
        labels=labels if isinstance(labels,tuple) else (labels,)
        for m in metrics:
            u=d.groupby(['unit','model'])[m].mean().unstack()
            delta=u['Chronos-2']-u['RC'];lo,hi=interval(delta)
            records.append(dict(zip(keys,labels))|dict(metric=m,n=len(delta),RC=u.RC.mean(),Chronos=u['Chronos-2'].mean(),delta=delta.mean(),lo95=lo,hi95=hi))
    return pd.DataFrame(records)

def ledgers():
    rows=[]
    for stage in ('baselines','confirmation','amovfly'):
        for path in (root/'results'/stage).rglob('complete.json'):
            v=json.loads(path.read_text());c=v.get('frozen_config',v.get('contract',v));m=c['model']
            if m not in ('RC','Chronos-2'):continue
            r=v['result'];s=r['ledger_s'][2];u=r['unavailable_fraction'];ds=np.asarray(r['selected_deadlines'])
            rows.append(dict(domain='实飞' if stage=='amovfly' else '合成',unit=str(c.get('world',c.get('flight_id'))),model=m,budget=c.get('budget',c.get('condition')),
                       seed=c.get('seed',0),G=r['G_mbps'],narrow=r['narrow_good_data_s'],time=s,rate=r['G_mbps']*180/s,obs=r['updates'],hold=ds@[.5,1,2,3,5,8,10]/ds.sum(),unavailable=100*u,
                       cond_geom=100*(r['R']-u)/(1-u)))
    a=pd.DataFrame(rows); cs=[]
    for (domain,budget),d in a.groupby(['domain','budget']):
        j=d[d.model=='Chronos-2'].merge(d[d.model=='RC'],on='unit',suffixes=('c','r'))
        ct=(j.timec-j.timer)*(j.ratec+j.rater)/360;cr=(j.ratec-j.rater)*(j.timec+j.timer)/360
        assert np.allclose(ct+cr,j.Gc-j.Gr)
        cs.append(dict(domain=domain,budget=budget,time_contribution=ct.mean(),rate_contribution=cr.mean(),G_gap=(ct+cr).mean()))
    return comparison(a,['domain','budget'],['G','time','rate','narrow','hold','obs','unavailable','cond_geom']),pd.DataFrame(cs)

def main():
    anchors=read_shards('anchors'); beams=read_shards('beam'); service=read_shards('service')
    assert len(anchors)==(640+59)*10 and not anchors.duplicated(['domain','unit','user','model','h']).any()
    assert len(beams)==20 and not beams.duplicated(['domain','unit','model']).any()
    np.testing.assert_allclose(beams.G32+beams.Gother,beams.replay_G,rtol=1e-12)
    beams['delta_G_vs_archive']=beams.replay_G-beams.original_G
    beams['delta_R_pp_vs_archive']=100*(beams.replay_R-beams.original_R)
    anchor=anchors[anchors.h>0].copy();anchor['cert32']=anchor.cert32.astype(float)
    keys=['position_m','angle_deg','safe32_prefix','cert32','az_halfwidth_deg','el_halfwidth_deg']
    summary=pd.concat([comparison(anchors[anchors.h==0],['domain','h'],['position_m','angle_deg']),comparison(anchor,['domain','h'],keys)],ignore_index=True)
    service['cert32']=service.cert32.astype(float)
    original=service[service.center==service.width].rename(columns={'center':'model'})
    both=comparison(original,['domain','h'],['err_m','err_deg','cert32'])
    service['false_accept']=service.cert32.astype(bool) & ~service.safe32_prefix
    cross=service.groupby(['domain','h','center','width','unit'])[['cert32','safe32_prefix','false_accept']].mean().groupby(level=[0,1,2,3]).mean().reset_index()
    beam_summary=comparison(beams,['domain'],['G32','Gother','replay_G','original_G'])
    formal,decomp=ledgers()
    desc=[
        ['比较方向','差值 delta 均为 Chronos-2 减 RC。两者均使用冻结的中心与各自校准管。'],
        ['闭环汇总','合成 40 世界（三个训练种子先在世界内平均），实飞 59 架次（种子 7）。T/M/L 服务感知额度合成 3/4.5/9 秒、实飞 0.1875/0.28125/0.5625 秒。0 dB MCS 余量，10 秒最长持有。'],
        ['共同预热','种子 7，在最后一次共同预热观测处发出一次预测。合成每世界 16 机，实飞每架次 1 机。实际观测完全共享，未来真值仅用于事后评分。'],
        ['闭环抽样','T 档种子 7，4 个合成世界（0/10/20/30），两类实飞航线各取清单内首、中、末架次。逐场重放的总吞吐与风险对照正式产物。'],
        ['服务期影子预测','在 RC 闭环中取每机服务时间第 20/70/120 秒后首次付费观测，给两个中心完全相同的历史。指标先在单世界或单架次内平均。交叉替换仅作离线准入敏感性，不重校准、不重跑交叉策略的吞吐。'],
        ['区间','配对单位为世界或架次，5,000 次重抽，RNG 271828。小样本重放区间为描述性结果。'],
        ['position_m / err_m','指定预测年龄处三维位置欧氏误差，米。angle_deg / err_deg 为环绕后的方位、俯仰误差向量的二范数，度。'],
        ['safe32_prefix / cert32','若一直使用 32×32，整段前缀在 20ms 真值网格的实际指向损失均不超过 3dB / 正式误差管前沿允许该期限，均表示比例（0–1）。'],
        ['az/el_halfwidth_deg','在给定期限端点的校准方位/俯仰半宽，度；前沿准入还检查整个前缀和相邻点包络。'],
        ['narrow / time / rate','有效 32×32 时间（秒）/ 有动作的实际数据发送时间（秒）/ 该时段内的净速率（Mbps）。'],
        ['G32 / Gother','对应阵列实际交付的比特除以完整 180 秒（Mbps）。两项相加是总吞吐；按阵列分账与按时间和速率分解是两种不同视角。'],
        ['false_accept','整组锚点中，误差管放行该窄束且至少一个前缀评分点实际超过 3 dB 的占比。'],
        ['hash_match','重算首个位置预测数组与对应正式种子 7 运行的 SHA256 是否逐字节一致。输入一致性与浮点输出哈希分别核对。']
    ]
    with pd.ExcelWriter(out/'RC与Chronos-2_差距证据.xlsx',engine='openpyxl') as wr:
        for name,frame in [('阅读说明',pd.DataFrame(desc,columns=['口径','说明'])),('正式闭环',formal),('吞吐差分解',decomp),('共同预热预测',summary),('服务期同历史预测',both),('中心与管的交叉准入',cross),('按波束交付',beam_summary),('预热锚点明细',anchors),('重放场次明细',beams),('服务锚点明细',service)]:
            frame.to_excel(wr,sheet_name=name,index=False);ws=wr.sheets[name];ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
            for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17545A')
            for j,col in enumerate(frame.columns,1):ws.column_dimensions[get_column_letter(j)].width=min(100,max(16,len(col)+4)) if name!='阅读说明' else [29,145][j-1]
    for title,frame in [('共同历史',summary),('服务期',both),('交叉',cross),('分束',beam_summary)]:
        print(title);print(frame.round(3).to_string(index=False))
    print('hashes',anchors.groupby(['domain','model']).hash_match.mean())
if __name__=='__main__':main()
