"""Audit and publish paired results using the world as the resampling unit."""
import json
from pathlib import Path
from html import escape
import hashlib
import itertools
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

OUT=Path(__file__).resolve().parent
ARMS=['C-native','C-special','CR-native','CR-transfer','CR-special','CRG-transfer','CRG-special','RC-special',
      'VECTOR-base','VECTOR-transfer','VECTOR-refit','RAFT-base','RAFT-transfer','RAFT-refit']
C_PAIRS=[('CR','C'),('CRG','CR'),('CRG','RC'),('VECTOR+P','VECTOR'),('RAFT+P','RAFT')]
N_PAIRS=[('C-special','C-native'),('CR-native','C-native'),('CR-transfer','C-special'),('CR-special','CR-transfer'),('CR-special','C-special'),
         ('CRG-transfer','CR-special'),('CRG-special','CRG-transfer'),('CRG-special','CR-special'),('CRG-special','RC-special')]
for model in ('VECTOR','RAFT'):
    N_PAIRS += [(model+'-transfer',model+'-base'),(model+'-refit',model+'-transfer'),(model+'-refit',model+'-base')]
H=[.5,1,2,3,5,8,10]; B=['T','M','L']
LABEL={'C-native':'Chronos · 无路线／原生管','C-special':'Chronos · 无路线／专用管','CR-native':'Chronos＋路线／原生管',
       'CR-transfer':'Chronos＋路线／原专用管','CR-special':'Chronos＋路线／重训专用管','CRG-transfer':'Chronos＋路线＋IMM／沿用路线专用管',
       'CRG-special':'Chronos＋路线＋IMM／融合专用管','RC-special':'RC 主线／专用管'}
for m in ('VECTOR','RAFT'):
    LABEL.update({m+'-base':m+' · 无新增路线／专用管',m+'-transfer':m+'＋路线／原专用管',m+'-refit':m+'＋路线／重训专用管'})
def ci(a):
    n=len(a);idx=np.random.default_rng(71023).integers(n,size=(5000,n))
    return np.quantile(np.asarray(a)[idx].mean(1),[.025,.975])
def contrasts(df,column,by,pairs,metrics):
    rows=[];keys=[by] if isinstance(by,str) else by
    for group,block in df.groupby(keys,sort=False):
        group=group if isinstance(group,tuple) else (group,)
        for a,b in pairs:
            aa=block[block[column]==a].set_index('world').sort_index()
            bb=block[block[column]==b].set_index('world').sort_index()
            assert len(aa) and aa.index.equals(bb.index)
            for metric in metrics:
                d=aa[metric]-bb[metric];lo,hi=ci(d)
                rows.append([*group,a,b,metric,len(d),aa[metric].mean(),bb[metric].mean(),d.mean(),lo,hi,int((d>0).sum())])
    return pd.DataFrame(rows,columns=[*keys,'方案','参照','指标','世界数','方案均值','参照均值','差','下界95','上界95','差大于零的世界数'])
def tables():
    c=pd.concat([pd.read_csv(OUT/f) for f in ['中心逐机.csv','VECTOR_中心逐机.csv','RAFT_中心逐机.csv']],ignore_index=True)
    h=pd.concat([pd.read_csv(OUT/f) for f in ['固定窄束保持逐机.csv','VECTOR_固定窄束保持逐机.csv','RAFT_固定窄束保持逐机.csv']],ignore_index=True)
    assert len(c)==40*16*7*10 and not c.duplicated(['method','world','uid','horizon_s']).any()
    assert len(h)==40*16*10 and not h.duplicated(['method','world','uid']).any()
    c['safe32_pct']=c.safe32*100;h['censored_pct']=h.censored_at_10*100
    cm=['position_m','angle_deg','safe32_pct']
    cw=c.groupby(['world','method','horizon_s'],as_index=False)[cm].mean()
    centers=c.groupby(['method','horizon_s'])[cm].mean().reset_index()
    p90=c.groupby(['method','horizon_s'])[['position_m','angle_deg']].quantile(.9).add_suffix('_p90').reset_index()
    centers=centers.merge(p90,on=['method','horizon_s']); cd=contrasts(cw,'method','horizon_s',C_PAIRS,cm)
    hw=h.groupby(['world','method'],as_index=False)[['safe_hold_32_s','censored_pct']].mean()
    hw['波束']='32×32';hd=contrasts(hw,'method','波束',C_PAIRS,['safe_hold_32_s','censored_pct'])
    net=[]
    native_diffs=[]; fingerprints={}
    for f in [*(OUT/'cells').glob('*.json'),*(OUT/'other_cells').glob('*.json')]:
        d=json.loads(f.read_text());r=d['result'];a=d['arm'];w=d['world'];b=d['budget']
        assert d['mcs_margin_db']==0 and d['seed']==7 and d['namespace']=='confirm_0db_20261122/v1'
        assert r['probes_per_user']==9000 and len(r['bits'])==16
        assert abs(sum(r['ledger_s'])-180)<1e-6 and abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-7
        counts=np.array(r['selected_deadlines']);n=counts.sum()
        assert 0<n<=r['updates']
        net.append([a,w,b,r['G_mbps'],100*r['R'],100*r['unavailable_fraction'],100*(r['R']-r['unavailable_fraction']),
                    counts@H/n,r['hold_mean_s'],r['hold_p50_s'],r['hold_p90_s'],r['narrow_good_data_s'],r['ledger_s'][0],
                    r['ledger_s'][0]/1.8,r['updates'],r['hold_segments'],100*n/r['updates'],np.quantile(np.array(r['bits'])/180e6,.1)])
        if a=='C-native' and 'reference_delta_mbps' in r:native_diffs.append(r['reference_delta_mbps'])
        fingerprints.setdefault((w,a),set()).add(r['initial_center_sha256'])
    assert len(native_diffs)==30
    assert set(map(tuple,[(r[0],r[1],r[2]) for r in net]))==set(itertools.product(ARMS,range(10),B)) and len(net)==420
    for w in range(10):
        for members in [('C-native','C-special'),('CR-native','CR-transfer','CR-special'),('CRG-transfer','CRG-special'),('VECTOR-transfer','VECTOR-refit'),('RAFT-transfer','RAFT-refit')]:
            assert len(set.union(*(fingerprints[w,m] for m in members)))==1
    net=pd.DataFrame(net,columns=['arm','world','budget','净吞吐Mbps','总风险%','无动作%','几何失准%','选定期限s','实际连续保持s','场内保持中位数s','场内保持P90s','有效窄束s','感知s','感知占比%','更新次数','保持段数','更新有动作%','逐机P10Mbps'])
    m=list(net.columns[3:]);ns=net.groupby(['budget','arm'],as_index=False)[m].mean();nd=contrasts(net,'arm','budget',N_PAIRS,m)
    return centers,cd,hw,hd,ns,nd,net,c.groupby(['group','method','horizon_s'])[cm].mean().reset_index(),max(abs(v) for v in native_diffs)
def report():
    c,cd,h,hd,n,nd,net,groups,drift=tables()
    dest=OUT/'配对结果.xlsx';tmp=OUT/'配对结果.partial.xlsx'
    widths=pd.read_csv(OUT/'误差管验证.csv')
    assert len(widths)==98
    sheets={'误差管验证':widths,'中心指标':c,'中心配对差':cd,'固定窄束保持':h,'固定窄束保持差':hd,'三预算闭环':n,'闭环配对差':nd,'逐场数据':net,'中心按飞行类型':groups}
    with pd.ExcelWriter(tmp,engine='openpyxl') as book:
        for name,df in sheets.items():
            df.to_excel(book,sheet_name=name,index=False)
            ws=book.sheets[name];ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
            for cells in ws.columns:
                head=cells[0];head.fill=PatternFill('solid',fgColor='1A4359');head.font=Font(bold=True,color='FFFFFF')
                ws.column_dimensions[head.column_letter].width=min(32,max(14,max(len(str(v.value)) for v in cells[:30])+2))
                for cell in cells[1:]:
                    if isinstance(cell.value,float):cell.number_format='0.000'
    wb=load_workbook(tmp,read_only=True);assert wb['逐场数据'].max_row==421;wb.close();tmp.replace(dest)
    def tab(df):return df.to_html(index=False,escape=True,border=0,float_format=lambda z:f'{z:.3f}')
    def para(x):return '<p>'+escape(x)+'</p>'
    def heading(x): return '<h2>'+escape(x)+'</h2>'
    def delta(df,group,key,a,b,metric):
        v=df[(df[group]==key)&(df['方案']==a)&(df['参照']==b)&(df['指标']==metric)].iloc[0]
        return f'{v["差"]:+.2f} [{v["下界95"]:+.2f}, {v["上界95"]:+.2f}]'
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>body{max-width:1120px;margin:3em auto;font:16px/1.65 sans-serif;color:#182e38}td,th{text-align:right;padding:4px 9px;border-bottom:1px solid #ccd4d3}table{border-collapse:collapse;margin-bottom:1.4em}th{background:#e3eeee}h2{margin-top:2.4em}p{max-width:92ch}</style><title>显式路线记忆的配对实验</title><h1>显式路线记忆对预测中心、波束保持及净吞吐的影响</h1>']
    page+= [para('0 dB，训练种子 7。比较对象为最新微调 Chronos-2、VECTOR、RAFT 及 RC 参照。中心评测使用 40 个确认世界，每个世界 16 架飞机。闭环在预先指定的前 10 个确认世界进行，14 个方案各在三个预算计分，共 420 场。区间按物理世界配对、5,000 次重抽样计算，反映当前种子和运行协议的差异。')]
    for m0,m1,n0,n1 in [('C','CR','C-special','CR-special'),('VECTOR','VECTOR+P','VECTOR-base','VECTOR-refit'),('RAFT','RAFT+P','RAFT-base','RAFT-refit')]:
        page.append(para(f'{m0} 加路线后，5 秒位置误差差值为 {delta(cd,"horizon_s",5.,m1,m0,"position_m")} 米，10 秒为 {delta(cd,"horizon_s",10.,m1,m0,"position_m")} 米。路线中心和重训专用管整套方案在紧预算的净吞吐差值为 {delta(nd,"budget","T",n1,n0,"净吞吐Mbps")} Mbps，总风险差值为 {delta(nd,"budget","T",n1,n0,"总风险%")} 个百分点。'))
    page+= [heading('1　显式路线信息的接入位置'),para('三者收到同一个路线专家在最后一次实测时刻发出的未来 20 步三轴相对位置、三轴标准差和支持度。路线记忆随本机已到达的实测更新，维护路线片段并用滤波速度估计沿路线进度。直接路线臂读取该专家的预报；只有 CRG 另外融合 IMM 输出的未来中心。评估用的未来真值只用于计分。Chronos 将 7 个量作为已知未来协变量；最近 180 s 的三轴实测仍是目标历史，新增通道的历史留空。直接路线臂沿用微调权重，没有针对路线协变量再次微调。CRG 另取新的路线条件中心和 IMM 中心，由新 G1 门控按期限融合。门控上下文使用可部署的 P/I 诊断和观测间隔。'),
         para('VECTOR 由原速度编码器得到状态，把 7 个量标准化后线性投影到 256 维，作为第二 GRU 的逐步解码输入，原速度头输出并积分成位置。RAFT 保留原训练轨迹多尺度检索，用 7→3 投影给检索分支的 20 步输出增加路线项，再经过原融合层。两个模型的原主干和检索库均冻结，只在 960 条训练路线上训练新增无偏置投影，并用原验证角损失选点。训练 RAFT 投影时排除查询路线自身。两个新增投影在零初始化时逐项复现原中心和专用管输入。Chronos 使用原生协变量接口，另两个模型学习投影，因此不能把三者的增益直接解释为同一种注入机制的优劣。'),
         para('原专用管迁移臂冻结原树、PCA（如有）和原校准系数。重训臂沿用同样特征与选模流程，用另 240 条训练路线拟合 LightGBM、400 条独立路线做双角前缀校准。CRG-transfer 沿用 CR-special 的管；CRG-special 重新拟合，并增加门控权重、两个中心的分歧。原生管比较限于 Chronos 系列。')],
    flat=[]
    for v in page:flat.extend(v if isinstance(v,list) else [v])
    page=flat
    page+= [heading('2　固定历史上的中心精度与几何保持'),para('所有中心由完全相同的 180 秒预热记录预测，期间每机约每秒一次。位置为端点三维距离，角度为方位与俯仰误差的二范数，均先在世界内平均。safe32_pct 是端点 32×32 平面阵列的指向损失不超过 3 dB 的比例。固定几何保持从最后实测逐 20 ms 计至首次超过 3 dB，最长观察 10 秒；达到上限的样本作为右截尾处理。更换误差管在给定同一历史时保持同一个中心。'),
            tab(c[c.horizon_s.isin([1,5,10])]),tab(h.groupby('method')[['safe_hold_32_s','censored_pct']].mean().reset_index()),
            heading('3　误差管覆盖与宽度'),para('相同 400 条验证路线、1,200 个发布起点。覆盖要求从发布至指定期限的两个角轴全程同时落入包络，半宽取期限末两轴平均，单位为度。Chronos 的 5 秒原专用管在原中心上为 2.17°、覆盖 91.6%；将同一个头用于路线条件中心后增至 4.15°、覆盖 96.3%；重训后为 2.60°、覆盖 91.4%。因此沿用相同树及校准系数，也会因中心与分位数输入改变而给出明显更宽的管。比较有效窄束时间可以检查这一改变如何反映到闭环。'),tab(widths[widths['期限s'].isin([5,10])]),heading('4　闭环净吞吐、预算、风险与实际保持'),para('预热和服务各 180 s，16 架；每次实测 3.1 ms。T、M、L 依次允许占用 1.667%、2.5%、5% 的服务时间，分别约 3、4.5、9 s。相同 value 规则按预算选择束宽/期限，按到期顺序刷新，TDMA 服务；MCS 余量 0 dB，最大期限 10 s。每世界每机用 9,000 个公共探针计分。'),
            para('选定期限仅统计新观测发布后成功选出动作的事件。实际连续保持计同一架、同一预测起点、同一束宽在服务窗内连续有效的时间，遇到换束、换预测、到期或无动作即截断，服务边界也截断。它包含该机等待 TDMA 时隙的时间。有效窄束则只计实际分配的数据时隙内 32×32 波束且几何损失合格的秒数。各指标先算每场再对世界等权。总风险由无动作和有动作且指向损失超过 3 dB 组成。')]
    for b in B:
        sub=n[n.budget==b].set_index('arm').loc[ARMS].reset_index()
        table=sub[['arm','净吞吐Mbps','总风险%','无动作%','几何失准%','选定期限s','实际连续保持s','有效窄束s','感知s']].copy()
        table.arm=table.arm.map(LABEL);page+=['<h3>'+b+'</h3>',tab(table)]
    page+= [heading('5　分开归因'),para('“沿用原专用管”反映同一套树和校准在新中心上的表现，树收到的中心几何和模型诊断会随中心输出改变。“重训减迁移”固定中心算法，检验重新拟合树、PCA 和校准的整套影响。在闭环里不同动作会产生不同后续观测；中心准确率因此另外在相同历史上测量。'),
            tab(nd[(nd['指标'].isin(['净吞吐Mbps','总风险%','实际连续保持s']))&(nd.budget=='T')][['方案','参照','指标','差','下界95','上界95']]),
            heading('6　来源与核验'),para(f'每组都核对了逐机比特、180 s 服务账本、感知预算、探针数、期限上限。同一中心各管的预热中心指纹逐世界一致。重复运行的无路线 Chronos 原生管对原确认回执的最大吞吐绝对差为 {drift:.6f} Mbps。40 个世界的预热时间与观测字段重新核对，旧确认缓存浮点重建容差 1e-10。结果只覆盖本次合成确认场景；真实飞行、其他训练种子及路线条件 Chronos 再训练的效果另需测量。'),'</html>']
    (OUT/'实验分析报告.html').write_text(''.join(page))
    from datetime import datetime
    with (OUT.parent/'实时实验进度.txt').open('a') as f:print(datetime.now().strftime('%F %T'),'显式路线实验完成。420/420 闭环、40 世界同历史精度、固定窄束保持和配对置信区间均已核验；交付 微调Chronos路线对照/实验分析报告.html 与 配对结果.xlsx。',file=f)
if __name__=='__main__':report()
