from pathlib import Path
import json,hashlib,html
import numpy as np,pandas as pd
from core import ROOT,log
BUDGET=['T','M','L']
ORDER=['RC','geometry','progress','branch','joint','joint-no-past','joint-no-edge','Chronos-2','VECTOR','RAFT','CR-special','CRG-special','VECTOR+P','RAFT+P']
ARM=['RC','joint-transfer','joint-refit','Chronos-2','Chronos-native','VECTOR','RAFT']
LABEL={'RC':'原 RC','joint-transfer':'联合中心＋转移管','joint-refit':'联合中心＋重训管','joint':'联合中心','geometry':'当前段固定速度','progress':'当前段学习速度','branch':'多分支固定速度','joint-no-past':'联合屏蔽历史通过点','joint-no-edge':'联合屏蔽末端后继','Chronos-2':'Chronos-2 专用管','Chronos-native':'Chronos-2 原生管','VECTOR':'VECTOR 专用管','RAFT':'RAFT 专用管','CR-special':'Chronos＋路线','CRG-special':'Chronos＋路线＋IMM 门控','VECTOR+P':'VECTOR＋路线','RAFT+P':'RAFT＋路线'}
def ci(d):
    d=np.asarray(d);assert np.isfinite(d).all()
    boot=d[np.random.default_rng(2211).integers(len(d),size=(5000,len(d)))].mean(1)
    return [d.mean(),*np.quantile(boot,[.025,.975]),np.sum(d>0),len(d)]
def html_table(table):return table.to_html(index=False,float_format=lambda v:f'{v:.3f}',border=0,na_rep='—')
def run():
    cell=[]
    for f in sorted((ROOT/'cells').glob('*.json')):
        v=json.loads(f.read_text());r=v['result'];w,b,m=v['world'],v['budget'],v['arm']
        assert len(r['bits'])==16 and abs(sum(r['ledger_s'])-180)<1e-6
        assert abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-7 and r['probes_per_user']==9000
        cell.append(dict(world=w,budget=b,arm=m,G=r['G_mbps'],risk=100*r['R'],unavailable=100*r['unavailable_fraction'],
                         pointing=100*(r['R']-r['unavailable_fraction']),hold=r['hold_mean_s'],actual=r['ledger_s'][0],duty=r['ledger_s'][0]/1.8,
                         updates=r['updates'],narrow=r['narrow_good_data_s'],hash=r['initial_center_sha256']))
    df=pd.DataFrame(cell);assert len(df)==210 and not df.duplicated(['world','arm','budget']).any()
    assert set(df.world)==set(range(40,50)) and set(df.arm)==set(ARM)
    pivot=df.pivot(index=['world','budget'],columns='arm',values='hash')
    assert (pivot['joint-transfer']==pivot['joint-refit']).all() and (pivot['Chronos-2']==pivot['Chronos-native']).all()
    metrics=['G','risk','unavailable','pointing','hold','actual','duty','updates','narrow']
    g=df.groupby(['budget','arm'])[metrics].mean().reindex(pd.MultiIndex.from_product([BUDGET,ARM],names=['budget','arm'])).reset_index()
    pairs=[]
    for b in BUDGET:
        q=df[df.budget==b].set_index(['arm','world'])
        for a in ('joint-refit','joint-transfer'):
            for base in ARM:
                if a==base:continue
                for k in metrics:
                    delta=q.loc[a,k].sort_index()-q.loc[base,k].sort_index()
                    pairs.append((b,a,base,k,*ci(delta)))
    C=['均差','下限','上限','正差个数','独立单位数']
    pairs=pd.DataFrame(pairs,columns=['预算','模型','对照','指标',*C])
    results={};comp=[];decs=[];groups=[];covers=[]
    for split in ('test','world','flight'):
        d=pd.read_csv(ROOT/(split+'.csv'))
        assert d.id.nunique()=={'test':400,'world':10,'flight':59}[split] and d.arm.nunique()==17
        keys=['position_m','angle_deg','safe32','hold32_s']
        grouped=d.groupby(['arm','horizon'])[keys].mean()
        results[split]=grouped.reset_index()
        for h in (.5,1,2,3,5,8,10):
            sub=d[d.horizon==h].groupby(['arm','id'])[keys].mean()
            for base in ORDER:
                if base=='joint':continue
                for k in keys:
                    dd=sub.loc['joint',k]-sub.loc[base,k]
                    comp.append((split,h,base,k,*ci(dd)))
        f=d.groupby(['arm','horizon'])[['cover','half_az_deg','half_el_deg']].mean().dropna().reset_index()
        f['split']=split;covers.append(f)
        f=d[(d.arm.isin(ORDER))&(d.horizon.isin([5,10]))].groupby(['group','arm','horizon'])[keys].mean().reset_index()
        f['split']=split;groups.append(f)
        dec=pd.read_csv(ROOT/(split+'_decoder.csv'))
        s=dec.drop(columns=['id','anchor','group']).mean();s['split']=split;decs.append(s)
    comp=pd.DataFrame(comp,columns=['样本','期限','对照','指标',*C]);dec=pd.DataFrame(decs);covers=pd.concat(covers);groups=pd.concat(groups)
    ceilings=pd.concat([pd.read_csv(ROOT/(s+'_oracle.csv')).groupby(['horizon','oracle'])[['position_m','angle_deg']].mean().reset_index().assign(split=s) for s in ('test','world','flight')])
    from history_compare import historical
    old=historical(df)
    timing=pd.read_csv(ROOT/'单次刷新耗时.csv').groupby('arm').ms.agg(n='size',p50='median',p95=lambda x:x.quantile(.95),p99=lambda x:x.quantile(.99)).reset_index()
    report=ROOT/'结果分析报告.html';sheets={'闭环均值':g,'闭环配对区间':pairs,'闭环逐世界':df,'离线各期限':results['test'],'新世界各期限':results['world'],'AMOVFLY各期限':results['flight'],'中心配对区间':comp,'覆盖与半宽':covers,'按轨迹类型':groups,'解码器诊断':dec,'可达误差诊断':ceilings,'旧世界与新世界':old,'单次服务时延ms':timing}
    with pd.ExcelWriter(ROOT/'配对比较.xlsx',engine='openpyxl') as out:
        for k,table in sheets.items():
            table.to_excel(out,sheet_name=k,index=False);ws=out.sheets[k];ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=min(32,max(12,max(len(str(c.value)) for c in col[:30])+2))
    parts=['<!doctype html><meta charset="utf-8"><title>RC 沿线进度与后继选择实验</title><style>body{max-width:1120px;margin:40px auto;font:16px/1.65 system-ui;color:#182b34}h1,h2{color:#123f4d}table{border-collapse:collapse;font-size:14px;margin:1em 0}td,th{padding:5px 9px;border-bottom:1px solid #c5ced1}th{background:#eaf1f2}pre{white-space:pre-wrap;font:inherit}code{color:#284674}</style><h1>RC 沿线进度与后继选择实验</h1>']
    def para(v):parts.append('<p>'+html.escape(v)+'</p>')
    def title(v):parts.append('<h2>'+html.escape(v)+'</h2>')
    def table(d):parts.append(html_table(d.replace({'arm':LABEL,'对照':LABEL,'模型':LABEL})))
    def lookup(split,arm,h,k):return results[split].set_index(['arm','horizon']).loc[(arm,h),k]
    def diff(split,h,base,k):
        r=comp[(comp['样本']==split)&(comp['期限']==h)&(comp['对照']==base)&(comp['指标']==k)].iloc[0]
        return f"{r['均差']:+.3f}［{r['下限']:+.3f}，{r['上限']:+.3f}］"
    def cp(b,base,k,a='joint-refit'):
        r=pairs[(pairs['预算']==b)&(pairs['模型']==a)&(pairs['对照']==base)&(pairs['指标']==k)].iloc[0]
        return f"{r['均差']:+.2f}［{r['下限']:+.2f}，{r['上限']:+.2f}］"
    title('主要结果')
    for h in (5,10):
        para(f"新世界共同预热历史，{h} 秒位置误差从原 RC 的 {lookup('world','RC',h,'position_m'):.2f} m 到联合中心的 {lookup('world','joint',h,'position_m'):.2f} m；同一批观测下 Chronos-2 为 {lookup('world','Chronos-2',h,'position_m'):.2f} m。联合相对原 RC 配对差 {diff('world',h,'RC','position_m')} m，相对 Chronos-2 {diff('world',h,'Chronos-2','position_m')} m。")
    para('三档预算 T/M/L 下，联合中心＋重训管相对原 RC 的净吞吐差依次为 '+'、'.join(cp(b,'RC','G') for b in BUDGET)+' Mbps；对应总风险差为 '+'、'.join(cp(b,'RC','risk') for b in BUDGET)+' 个百分点。方括号为按世界配对重抽样的 95% 百分位区间。')
    from figure import figure
    parts.append(figure(results['world'],pairs))
    para('左图的中心曲线来自相同的预热观测；右图显示发布新中心后，沿用旧管和重训新管各自带来的净吞吐变化，误差线为同世界配对 95% 区间。')
    title('1　模型与实验设计')
    parts.append('<pre>'+html.escape((ROOT/'机制与验证方案.txt').read_text())+'</pre>')
    title('2　训练量、超参数选择与误差管校准')
    train=json.loads((ROOT/'training.json').read_text());rows=[]
    for m,configs in train.items():
        for c in configs:
            tr=c['trace'];rows.append([LABEL[m],c['width'],c['lr'],c['seed'],len(tr),np.argmin(tr)+1,min(tr)])
    table(pd.DataFrame(rows,columns=['机制','宽度','初始学习率','种子','实训轮数','原始验证曲线最低轮','最低损失']))
    head=json.loads((ROOT/'heads/joint/special.json').read_text())
    para(f"联合中心专用管沿用 RC 的因果诊断字段（例如两专家角差），选中 num_leaves、min_child_samples 和树数为 {head['chosen']}，共 {head['n_features']} 个输入。独立校准采用每条路线一个锚点，水平为 90%。离线验证选中心超参；独立 test 的覆盖和确认世界覆盖在后文分别给出。")
    for split,caption in [('test','3　独立离线轨迹'),('world','4　新世界同历史对比')]:
        title(caption)
        para(('400 条仿真路线，1200 个锚点，包括不同观测间隔。' if split=='test' else '10 个新确认世界，160 架次，每个用户在约 180 秒的共同预热观测末端发布一次。')+'本节固定预测起点，无在线感知/调度反馈；中心评分不读取误差管。位置和角度为端点均值，safe32 为此刻 32×32 波束损耗不超过 3 dB 的比例；hold32_s 是从发布到首次超阈的时长，10 秒截尾。')
        small=results[split];table(small[small.arm.isin(ORDER)&small.horizon.isin([5,10])].round(4))
        para(f"10 秒联合相对只学进度的位置差为 {diff(split,10,'progress','position_m')} m，相对只学分支为 {diff(split,10,'branch','position_m')} m。屏蔽过去通过点后的同权重干预差（联合减屏蔽）为 {diff(split,10,'joint-no-past','position_m')} m；屏蔽历史末端边时为 {diff(split,10,'joint-no-edge','position_m')} m。消融干预中候选概率重新归一化。")
    for split,label in [('test','离线测试'),('world','新确认世界')]:
        strat=groups[(groups.split==split)&(groups.horizon==10)].pivot(index='group',columns='arm',values='position_m')
        strat=strat[['RC','joint','Chronos-2','joint-no-past','joint-no-edge']].copy()
        strat['联合减原 RC (m)']=strat['joint']-strat['RC'];strat['联合减 Chronos (m)']=strat['joint']-strat['Chronos-2']
        para(label+'按轨迹类型分解。以下均是同历史下 10 秒预测的平均位置误差（米）；沿用原数据集的六类轨迹标签。')
        table(strat.reset_index())
    title('4.1　AMOVFLY 实飞轨迹的外部回放')
    para('沿用上轮筛选且冻结的 59 架次三角和四边形航线、同一站点和源文件 SHA-256，未用实飞数据调参。测得的位置轨迹与同期速度作为真值/传感器模拟输入。以每秒一条观测的固定回放比较全部中心，每架次取预热末端及约 240、300、360 秒处共 4 个锚点。这里固定频率、没有闭环刷新预算；无线指向损耗与感知噪声来自原协议，实飞数据未提供链路实测吞吐。')
    flight=results['flight']
    table(flight[flight.arm.isin(ORDER)&flight.horizon.isin([5,10])])
    para(f"实飞 10 秒，联合相对原 RC 位置差 {diff('flight',10,'RC','position_m')} m，相对 Chronos-2 {diff('flight',10,'Chronos-2','position_m')} m；固定 32×32 首次超阈保持相对原 RC {diff('flight',10,'RC','hold32_s')} s。区间按架次配对重抽样，四个锚点保留在同一架次内。")
    title('5　固定中心后的误差包络')
    para('双角联合前缀覆盖要求从发布到所报期限的所有 20 ms 样点两轴都处在包络内。两列半宽单位为度，分别对应方位和俯仰。Chronos 两管及联合两管共享各自的中心。所有测试覆盖比例均为实测值。')
    table(covers[covers.horizon.isin([5,10])])
    for split,label in [('test','离线测试'),('world','新确认世界')]:
        t=covers[(covers.split==split)&(covers.horizon==10)].set_index('arm')
        a,b=t.loc['joint-transfer'],t.loc['joint-refit']
        para(f"{label}固定联合中心，换管前后的 10 秒双角前缀覆盖为 {a.cover:.1%} 和 {b.cover:.1%}；方位半宽 {a.half_az_deg:.2f}° 和 {b.half_az_deg:.2f}°，俯仰半宽 {a.half_el_deg:.2f}° 和 {b.half_el_deg:.2f}°。两组中心逐样本相同，差异来自尺度函数与校准因子。")
    title('6　同世界闭环的预算、吞吐、风险和保持')
    para('16 用户，180 秒预热和 180 秒服务；T/M/L 三档预算、0 dB MCS、10 ms TDMA 和价值调度均固定。各模型可触发不同观测。G 单位 Mbps；risk、unavailable、pointing 和 duty 均为百分数。hold 是实际有效波束分段的均值（秒），actual 是该世界实际感知秒数，updates 是刷新次数，narrow 是有效数据时隙中使用 32×32 且指向损耗合格的秒数。每行对十个世界等权。')
    table(g)
    for b in BUDGET:
        para(f"{b} 档将旧管用于新中心，相对 RC 吞吐差 {cp(b,'RC','G','joint-transfer')} Mbps；在此新中心上重训管后再变化 {cp(b,'joint-transfer','G')} Mbps、风险变化 {cp(b,'joint-transfer','risk')} 个百分点。完整系统相对 Chronos 原生管吞吐差 {cp(b,'Chronos-native','G')} Mbps。")
    para('同一新中心下，转移管与重训管的起始中心指纹在 30 个世界×预算组合中逐一一致；进入闭环后由于刷新时刻不同，后续观测也会不同。三档预算下的配对结果说明，更适合新中心的尺度管决定了几何精度改善能兑现多少吞吐。')
    for b in BUDGET:
        t=g[g.budget==b].set_index('arm')
        para(f"{b} 档，旧 RC 到联合重训管的可用窄波束优质数据时间从 {t.loc['RC','narrow']:.2f} 秒到 {t.loc['joint-refit','narrow']:.2f} 秒；实际感知从 {t.loc['RC','actual']:.2f} 秒到 {t.loc['joint-refit','actual']:.2f} 秒。联合模型净吞吐相对 VECTOR 为 {cp(b,'VECTOR','G')} Mbps，相对 RAFT 为 {cp(b,'RAFT','G')} Mbps，相对 Chronos 专用管为 {cp(b,'Chronos-2','G')} Mbps。")
    title('7　上一轮和本轮世界的差异')
    para('只比较两轮原样复用的五套系统。两轮都是 0 dB、相同三档预算和调度，旧轮为世界 0–9，本轮为世界 40–49。本表是两个不同世界样本的均值对照，不作逐世界配对。判断新 RC 的提升仍使用本轮同世界原 RC 作基线。')
    table(old)
    title('8　机制诊断、适用范围与复核')
    table(dec)
    para('逐期限的事后可达性分析固定旧门控、联合信任度和预测起点，额外给评估器真实端点。oracle_branch 固定已预测里程，择优选候选曲线（含原软混合）；oracle_progress 固定已学分支权重，在 0–12h 米范围以 2 米步长择优选里程；oracle_both 同时择优。各列独立最小化本列误差，所得值用于定位剩余误差来源；这些事后行不参加通信比较。')
    table(ceilings)
    for spl,lab in [('test','离线测试'),('world','确认世界')]:
        t=ceilings[(ceilings.split==spl)&(ceilings.horizon==10)].set_index('oracle').position_m
        para(f"{lab}，10 秒已学中心为 {t['learned']:.2f} m。如果只允许事后改变分支，误差可少 {t['learned']-t['oracle_branch']:.2f} m；只允许事后改变里程，可少 {t['learned']-t['oracle_progress']:.2f} m。两项容许同时改变时为 {t['oracle_both']:.2f} m。各上限互有重叠，不能相加。")
    para('weight 为候选平均软权重；distance 是累计预测前进米数；effective 是信任度乘原门控后的实际修正系数。它们说明预测中实际使用哪些来源。历史通过点是新引入的独立候选，组合模型的收益不能全部归给原路线图的末端转移。每种轨迹类型见工作簿。')
    para('本轮最值得保留的是历史通过点和沿线进度的联合修正。后续优先在新训练划分上监督预测每个未来期限的沿线累积里程，并单列减速、停车、再起步及复访时的节奏错位；保持本轮原门控和误差管对照设计，另留新世界确认。事后可达性代表改进空间，实际能否学到需要下一轮验证。')
    title('9　服务器单用户刷新时延')
    para('矩阵完成后，在同一服务器、同一个新世界、同一串真实观测上逐个模型串行计时。各先预热整批与两轮单用户更新，再测 16 用户×8 次。包括从新观测更新模型、返回中心及误差尺度，GPU 在计时前后同步。计时不含传感、调度或无线传输，不能替代机载硬件测试。下表单位毫秒。')
    table(timing)
    para('数据来自仿真。10 个闭环世界的配对区间会随世界重抽样变化，各期限和各模型的探索性比较未作多重校正。风险同时受感知刷新和指向损耗影响，保持更长不一定带来更高吞吐。本实验没有把吞吐写入中心训练损失，也未测机载部署的端到端时延。')
    para('工作簿含所有期限、两套独立样本、原始闭环逐世界结果和配对区间。test.csv、world.csv 是逐锚点评分；cells 中有完整账本和起始中心指纹。')
    report.write_text('\n'.join(parts))
    log('实验验收：12 组训练曲线、400 条独立 test、10 个新世界共同历史、59 架次实飞轨迹共同历史、210/210 个闭环单元；两对共享中心的管跨世界和预算指纹一致，逐场核对 180 秒账本。配对比较.xlsx 和结果分析报告.html 已写入。')
if __name__=='__main__':run()
