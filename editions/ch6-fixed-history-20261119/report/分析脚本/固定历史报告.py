"""Typeset the common-history amendment using only the finalized paired result tables."""
import numpy as np
import pandas as pd


def one(df,**filters):
    for col,val in filters.items():
        df=df[np.isclose(df[col],val) if isinstance(val,(int,float)) else df[col]==val]
    assert len(df)==1, filters
    return df.iloc[0]


def signed(x):return f'{x:+.2f}'


def interval(r):return f'{r.delta:+.2f} [{r.lo:.2f}, {r.hi:.2f}]'


def conclusion(data):
    dd=data('固定历史工况全指标')
    changes=data('高成本两档交互')
    hist=data('固定历史与原协议配对')
    def at(c):return one(dd,cost_ms=c,budget_pct=5,scheduler='value',method='RC')
    effect=one(hist,cost_ms=12.4,budget_pct=5,scheduler='value',method='RC',metric='net_mbps')
    gain=one(changes,protocol='密集',arm='RC-I',metric='net_mbps')
    verdict=('该区间跨零，因此原报告中明确的优势回落不能直接归因于服务成本。' if gain.lo<=0<=gain.hi else '该区间给出了控制初始历史后增益变化的方向。')
    return (f'把全部方法的起点统一为每架 179–180 次完全相同的真实观测后，RC 在 9.3 和 12.4 ms 下分别达到 '
            f'{at(9.3).net_mbps:.2f} 和 {at(12.4).net_mbps:.2f} Mbps。12.4 ms 服务成本固定时，仅将历史从该档的稀疏预热改为密集预热，'
            f'RC 改变 {interval(effect)} Mbps。统一历史下从 9.3 到 12.4 ms，RC−I 改变 {interval(gain)} Mbps。'+verdict)


def pressure_figure(hist,pairs,destination):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.2,2.8),layout='constrained')
    costs=[3.1,6.2,9.3,12.4]
    for m,color in [('I','#747e85'),('RC','#007c78')]:
        points=[one(hist,cost_ms=c,budget_pct=5,scheduler='value',method=m,metric='net_mbps') for c in costs]
        axes[0].plot(costs,[r.left for r in points],'o-',c=color,label=m+' 密集')
        axes[0].plot(costs,[r.right for r in points],':',c=color,label=m+' 原协议')
    axes[0].set_ylabel('全网吞吐（Mbps）');axes[0].legend(frameon=False,fontsize=8,ncol=2)
    for m,color in [('RC','#007c78'),('I','#747e85')]:
        points=[one(hist,cost_ms=c,budget_pct=5,scheduler='value',method=m,metric='net_mbps') for c in costs]
        axes[1].errorbar(costs,[r.delta for r in points],yerr=[[r.delta-r.lo for r in points],[r.hi-r.delta for r in points]],fmt='o-',c=color,capsize=3,label=m)
    axes[1].axhline(0,c='gray',lw=.7);axes[1].legend(frameon=False,fontsize=8)
    axes[1].set_ylabel('密集历史减原协议（Mbps）')
    for ax in axes:
        ax.set_xticks(costs);ax.set_xlabel('服务阶段单次感知成本（ms）');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(destination,dpi=200);plt.close(fig)


def append_pages(data,page,sub,para,table,add_pressure_figure):
    dd=data('固定历史工况全指标');orig=data('测试工况全指标')
    pairs=data('固定历史方法配对'); hist=data('固定历史与原协议配对'); changes=data('高成本两档交互')
    effects=data('历史及服务效应');match=data('固定历史等间隔对照');sched=data('固定历史调度对照')
    sf=data('同动作F预算对照');errors=data('起点预测密集减原历史')
    def avg(c,b,m,s='value',d=dd):return one(d,cost_ms=c,budget_pct=b,scheduler=s,method=m)
    def hw(c,b,m,metric='net_mbps'):return one(hist,cost_ms=c,budget_pct=b,method=m,scheduler='value',metric=metric)
    def gain(c,b):return one(pairs,cost_ms=c,budget_pct=b,arm='RC',ref='I',metric='net_mbps')
    def swing(m,p):return one(changes,protocol=p,arm=m,metric='net_mbps')
    def B(b):return '1.667' if b<2 else f'{b:g}'
    page('补充设计：用同一份观测历史进入服务')
    para('原实验回答一整次任务从预热到服务都处于给定资源条件时会怎样。现在将问题细分为两次同世界比较：所有组先得到同一密集历史，只改变服务阶段成本或预算；在同一服务设置下，再比较密集历史和该档原历史。后一对比包含预热观测的数量、间距和末次观测新鲜度的共同作用。')
    table(['比较','原实验是否受起始历史影响','本次处理'],[
        ['固定 5% 扫成本','成本同时作用于预热和服务','3.1、6.2、9.3、12.4 ms 共用 3.1 ms/5% 历史，I 和 RC 各 30 个世界。'],
        ['固定 3.1 ms 扫共享预算','预算同时作用于预热和服务','5%、2.5%、5%/3 共用密集历史，I/P/RC/F 各 30 世界，C/CR_raw 各取相同 10 世界。'],
        ['同点方法及调度','同工作点原本已共用观测协议。服务开始后动作会自然分叉','沿用 frozen full 模型、校准和配对世界。紧预算下重测 RC 的 value/EDF/uniform。'],
        ['固定方案 F','原验证按预算选动作：32×32，2/2/3 秒','同预算跨历史冻结原 F 动作；另用在三档预算都可行的 32×32、3 秒比较预算本身。'],
        ['等间隔配对及验证消融','等间隔对照本来就同时改两个量；同工况验证门控或校准已共用起点','等间隔作为两种干预的对比。验证选型冻结；完整方法差包含自身预测与校准特性。']],[.23,.35,.42])
    para('复用完全相同协议的基点 140 场，新跑原矩阵另外 520 场、同动作 F 另 60 场；另重跑一场基点 I 验证数值。历史哈希涵盖时间戳、原始量测、位置、协方差与径向速度观测。新跑的每场都与所属世界的独立密集起点重建逐项哈希一致。每世界全站 2,872 次，前 8 架各 180 次、后 8 架各 179 次。')
    para('服务期继续执行闭环：每个方法自行决定何时再观测。强制服务阶段也拥有同一条后续观测轨迹会改变此处的研究问题。密集预热的额外资源在服务开始前发生，全网吞吐的分母仍为 180 秒服务时间。')
    page('控制起始历史后，成本效应与历史效应各有多大')
    add_pressure_figure(hist,pairs)
    table(['服务成本','I 原 / 密集','RC 原 / 密集','密集 RC−I，95% 区间','RC 密集−原，95% 区间'],
          [[f'{c:g} ms',f'{avg(c,5,"I",d=orig).net_mbps:.1f} / {avg(c,5,"I").net_mbps:.1f}' if c!=12.4 else f'{hw(c,5,"I").right:.1f} / {avg(c,5,"I").net_mbps:.1f}',
            f'{hw(c,5,"RC").right:.1f} / {hw(c,5,"RC").left:.1f}',interval(gain(c,5)),interval(hw(c,5,'RC'))] for c in [3.1,6.2,9.3,12.4]],
          [.13,.19,.19,.245,.245])
    para('全部为 Mbps，30 个世界配对。第一行复用同一批基点数据，历史差按设计等于零。把“密集减原”读作在同一成本、同一策略下更换预热协议造成的整场净效应。')
    table(['9.3 → 12.4 ms','原全生命周期变化','固定密集历史变化','两种变化之差'],[[m,interval(swing(m,'全生命周期')),interval(swing(m,'密集')),interval(swing(m,'密集减全生命周期的变化'))] for m in ['I','RC','RC-I']],[.14,.29,.28,.29])
    a=swing('RC-I','密集'); hh=swing('RC','密集减全生命周期的变化')
    trend=('仍有明确回落' if a.hi<0 else '在统一历史后反而增加' if a.lo>0 else '变化区间包含零，统一历史下回落方向不确定')
    para(f'原报告的 −67.04 Mbps 描述全生命周期 RC−I 的变化。控制历史后该差值{trend}。对 RC 自身，把两档都改成相同密集起点后，12.4 相对 9.3 ms 的变化量与旧值相差 {interval(hh)} Mbps。这是两种预热协议对成本敏感性的交互，不能拆成互相独立的机制百分比。')
    x,y=avg(9.3,5,'RC'),avg(12.4,5,'RC')
    para(f'在同一密集起点上，RC 的服务期观测由 {x.observations_service:.1f} 变为 {y.observations_service:.1f} 次，32×32 窄束时间 {x.narrow_pct:.2f}% → {y.narrow_pct:.2f}%，3 秒以上预测年龄 {x["age_3+_pct"]:.2f}% → {y["age_3+_pct"]:.2f}%。每场数据期秒数为 {x.data_s:.2f} / {y.data_s:.2f}，数据期间有效速率为 {x.rate_in_data_mbps:.2f} / {y.rate_in_data_mbps:.2f} Mbps。')
    page('共享预算、Chronos 与调度的复核')
    table(['服务预算 %','I','P','RC','原动作 F','密集 RC−I，95% 区间'],[[B(b),*[f'{avg(3.1,b,m).net_mbps:.2f}' for m in ['I','P','RC','F']],interval(gain(3.1,b))] for b in [5,2.5,5/3]],[.14,.14,.14,.14,.15,.29])
    para('以上为统一密集历史下的全网 Mbps。原动作 F 在同预算的两种历史下保持同一周期，跨预算则沿用原验证选定的周期。')
    b=5/3
    gain_change=one(effects,arm='RC-I',contrast='密集减原',metric='net_mbps',cost_ms=3.1,budget_pct=b)
    rc_h=hw(3.1,b,'RC');i_h=hw(3.1,b,'I')
    para(f'最紧预算时把预热固定为密集，RC 和 I 各自相对原结果改变 {rc_h.delta:+.2f}、{i_h.delta:+.2f} Mbps；因此 RC−I 相对初版的 +234.03 Mbps 改变了 {interval(gain_change)} Mbps。直接比较两方法的历史收益可以避免把 I 同样受益的部分归给路线融合。')
    cm=sf[sf.metric=='net_mbps'].sort_values('budget_pct',ascending=False)
    para('同时冻结 F 的束宽和周期（32×32、3 s），5%、2.5%、1.667% 的吞吐分别为 '+'、'.join(f'{r.left:.2f}' for r in cm.itertuples())+' Mbps。最紧与最松预算的同世界吞吐差是 '+interval(cm.iloc[-1])+' Mbps。相同周期使较松预算可能用不满。')
    rows=[]
    for b in [5,2.5,5/3]:
        for m in ['C','CR_raw']:
            g=pairs[(pairs.metric=='net_mbps') & (pairs.arm==m) & np.isclose(pairs.budget_pct,b)].set_index('ref')
            rows.append([B(b),m,*[signed(g.loc[k,'delta']) for k in ['I','P','RC','F']]])
    table(['预算 %','Chronos','减 I','减 P','减 RC','减 F'],rows)
    para('Chronos 对照仅使用共同的前 10 个世界，数值为 Mbps。同点对照识别整套方法差；各方法都有专用中心和尺度头。')
    z=one(pairs,arm='CR_raw',ref='RC',metric='net_mbps',cost_ms=3.1,budget_pct=5/3)
    para(f'同一紧预算下，CR_raw−RC 的固定历史结果为 {interval(z)} Mbps，原全生命周期为 −136.86 Mbps。初始观测已配平，差值仍包括两套预测中心及各自尺度头对调度的共同影响。')
    ordered=[one(pairs,arm='CR_raw',ref='C',metric='net_mbps',cost_ms=3.1,budget_pct=b) for b in (2.5,5/3)]
    para(f'原协议下两档 CR_raw 的平均吞吐均高于 C。固定历史后，2.5% 的 CR_raw−C 为 {interval(ordered[0])} Mbps，区间跨零；1.667% 为 {interval(ordered[1])} Mbps，排序反转。这条原结论具有预热协议依赖性。')
    rows=[[s,f'{avg(3.1,5/3,"RC",s).net_mbps:.2f}',f'{avg(3.1,5/3,"RC",s).hold_fail_pct:.2f}',f'{avg(3.1,5/3,"RC",s).worst10_mbps:.2f}'] for s in ['value','edf','uniform']]
    table(['RC 调度','净吞吐 Mbps','保持失效 %','最差两机 Mbps'],rows)
    para('紧预算 value−EDF 净吞吐为 '+interval(one(sched,metric='net_mbps',ref='edf'))+' Mbps；保持失效差为 '+interval(one(sched,metric='hold_fail_pct',ref='edf'))+' 个百分点。')
    page('起点预测与结论适用范围')
    para('对 30 个测试世界分别恢复六种预热协议的最终预测包，冻结 G1 并停止新观测。各方法预测相同绝对目标时刻，即服务开始之后 1、3、5 秒。先在世界内对 16 架的方位和俯仰角误差计算 RMSE，再对世界等权。这样的边界预测同时包含末次观测新鲜度和历史学习质量。')
    rows=[]
    for dt in [1,3,5]:
        rr=one(errors,horizon_s=dt,method='RC',cost_ms=12.4,budget_pct=5)
        ii=one(errors,horizon_s=dt,method='I',cost_ms=12.4,budget_pct=5)
        rows.append([str(dt),f'{rr.left:.3f} / {rr.right:.3f}',interval(rr),f'{ii.left:.3f} / {ii.right:.3f}',interval(ii)])
    table(['服务开始后 s','RC 密集 / 稀疏 °','RC 差，95% 区间','I 密集 / 稀疏 °','I 差，95% 区间'],rows,[.14,.2,.23,.2,.23])
    para('这里稀疏指 12.4 ms/5% 预热。固定目标时刻的起点误差可直接验证“历史协议会影响预测”；它不能直接换算成整场吞吐的百分比，也不能把共同跟踪状态与路线记忆单独拆开。')
    sub('等刷新间隔与低尾可靠性')
    for r in match.itertuples():
        para(f'统一起点后，{r.cost_ms:g} ms/5% 对 3.1 ms/{B(r.match_budget_pct)}% 的 RC−I 增益分别为 {r.left:.2f} 与 {r.right:.2f} Mbps，两种干预之差 {interval(r)} Mbps。该项仍同时改变 c 与 ρ。')
    c=avg(12.4,5,'RC');i=avg(12.4,5,'I')
    para(f'12.4 ms、统一密集历史下，RC 与 I 的最差两机平均吞吐为 {c.worst10_mbps:.2f} / {i.worst10_mbps:.2f} Mbps，保持段几何失效为 {c.hold_fail_pct:.2f}% / {i.hold_fail_pct:.2f}%。')
    runs=data('合并逐场指标2080')
    times=runs[(runs.batch=='固定历史') & (runs.method=='RC') & (runs.cost_ms==12.4)]
    slow=times[times.world==22].iloc[0]
    para(f'实现耗时还暴露了一个极端世界。在 12.4 ms 密集历史 RC 中，世界 22 的单次最长更新实耗 {slow.max_update_cpu_s:.1f} 秒，各世界单次最长更新的中位数为 {times.max_update_cpu_s.median():.2f} 秒。同配置运行的栈采样反复落在线路片段分支展开。仿真将推理时延设为零，因此该实耗没有计入上表通信吞吐；实际部署需要另测在线时限。')
    sub('如何复核')
    para('原 1,500 场加 580 场新的科学比较，共 2,080 场唯一记录，另 1 场数值复核独立记账；基点复用及方案之间的统计复用均不增加样本数。完整记录、两种预热、同动作 F、验证/测试划分和原始 metrics 路径见“合并逐场指标2080.csv”。工作簿按协议分表，未把不同预热混合求均值。每场输入指纹和启动预测的逐机误差分别见“固定历史逐场指纹.csv”和“服务起点预测逐机.csv”。')
