"""Render the Word and PDF reports from shared text, figures and audited tables."""
from pathlib import Path
import ast
import json
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

root=Path(__file__).resolve().parents[1]
tab=root/'数据表'
def data(name): return pd.read_csv(tab/(name+'.csv'))
avg=data('测试工况全指标'); pairs=data('测试方法配对'); valid=data('验证工况全指标')
primary=root/'主实验归档/runs/paired600'
font='/data/chen/teacher_report_20261118/assets/NotoSansSC-Regular.ttf'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
pp=pairs.query("budget_pct==5 and metric=='net_mbps' and arm=='RC' and ref=='I'").sort_values('cost_ms')
fig,ax=plt.subplots(1,2,figsize=(9.2,3.05),layout='constrained')
for method,color in [('I','#757e88'),('RC','#007c78')]:
    g=avg.query("budget_pct==5 and scheduler=='value' and method==@method").sort_values('cost_ms')
    ax[0].plot(g.cost_ms,g.net_mbps,marker='o',label=method,color=color,lw=2)
ax[0].set_ylabel('全网净吞吐（Mbps）'); ax[0].legend(frameon=False)
ax[1].errorbar(pp.cost_ms,pp.delta,yerr=[pp.delta-pp.lo,pp.hi-pp.delta],fmt='o-',capsize=4,color='#854721')
ax[1].axhline(0,lw=.7,color='gray');ax[1].set_ylabel('RC − I 配对差（Mbps）')
for a in ax:
    a.set_xticks(pp.cost_ms);a.set_xlabel('单次全站感知成本（ms）');a.grid(axis='y',alpha=.2);a.spines[['right','top']].set_visible(False)
fig.savefig(root/'图/压力曲线.png',dpi=210);plt.close(fig)

doc=Document(); sec=doc.sections[0];sec.page_height=Cm(29.7);sec.page_width=Cm(21);sec.top_margin=Cm(1.9);sec.bottom_margin=Cm(1.8);sec.left_margin=sec.right_margin=Cm(2)
for name,size in [('Normal',10),('Title',24),('Heading 1',17),('Heading 2',12)]:
    st=doc.styles[name];st.font.name='Microsoft YaHei';st.font.size=Pt(size);st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑')
    st.paragraph_format.space_after=Pt(5)
sec.header.paragraphs[0].text='RouteClock  /  共享感知预算闭环实验'
foot=sec.footer.paragraphs[0];foot.alignment=2;foot.add_run('2026 年 11 月 19 日    ·    ')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
pdfmetrics.registerFont(TTFont('CJK',font))
styles={name:ParagraphStyle(name,fontName='CJK',fontSize=size,leading=leading,textColor=colors.HexColor(color),wordWrap='CJK',spaceAfter=gap)
        for name,size,leading,color,gap in [('title',21,31,'#123b46',12),('h1',16,25,'#123b46',10),('h2',11,18,'#16655f',4),
            ('body',9.8,16,'#203039',7),('cell',8.4,12,'#203039',0),('head',8.4,12,'#ffffff',0)]}
story=[]
def para(text,style='body'):
    doc.add_paragraph(text,style={'body':'Normal','title':'Title','h1':'Heading 1','h2':'Heading 2'}[style])
    story.append(Paragraph(escape(text),styles[style]))
def page(title):doc.add_page_break();story.append(PageBreak());para(title,'h1')
def sub(s):para(s,'h2')
def table(header,rows,widths=None):
    vals=[header]+[[str(s) for s in row] for row in rows]
    t=doc.add_table(rows=0,cols=len(header));t.style='Light Shading Accent 1';t.alignment=WD_TABLE_ALIGNMENT.CENTER
    if widths:
        t.autofit=False
        for col,w in zip(t.columns,widths):col.width=Cm(w*17)
    for i,row in enumerate(vals):
        cells=t.add_row().cells
        for j,(cell,s) in enumerate(zip(cells,row)):
            if widths:cell.width=Cm(widths[j]*17)
            p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
            run=p.add_run(s);run.bold=i==0;run.font.size=Pt(9)
    story.append(Table([[Paragraph(escape(s),styles['head' if i==0 else 'cell']) for s in row] for i,row in enumerate(vals)],
                       colWidths=[x*478 for x in (widths or [1/len(header)]*len(header))],repeatRows=1,
                       style=TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#19545e')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#eef4f3')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)])))
    story.append(Spacer(1,7))
def fig1():
    f=root/'图/压力曲线.png';doc.add_picture(str(f),width=Cm(16.5));story.append(Image(str(f),width=465,height=154))
def f(x):return f'{x:.2f}'
def ci(r):return f'{r.delta:+.2f} [{r.lo:.2f}, {r.hi:.2f}]'
def budget(x):return '1.667' if x<2 else f'{x:g}'
def row_at(cost,bud,met,s='value'):
    return avg[(avg.cost_ms==cost)&np.isclose(avg.budget_pct,bud)&(avg.method==met)&(avg.scheduler==s)].iloc[0]

para('共享感知预算下的 RouteClock','title')
para('第六章代码平台、验证设计与闭环结果  |  2026 年 11 月 19 日','h2')
para('本报告回答两个问题：在 16 架飞行器共用感知时间时，路线融合 RC 什么时候有通信收益？这种收益以多少感知时间、波束保持失效和弱用户吞吐为代价？正文采用归档后的逐世界输出，分别报告验证集选型、原计划独立测试和测试启动后登记的压力扩展。')
sub('最值得保留的三个结果')
para('在 3.1 ms、5% 的宽松点，RC 比惯性模型 I 高 6.03 Mbps，约为 I 的 0.49%。单次成本升至 9.3 ms 后，优势达到 230.24 Mbps；保持 3.1 ms、把预算压到 1.667% 也得到 234.03 Mbps。用两种方式降低刷新机会均能显现路线融合的价值。')
para('12.4 ms、5% 时，RC 的绝对吞吐降至 638.86 Mbps，仍比 I 高 163.19 Mbps；相对 9.3 ms，两者的吞吐差收窄 67.04 Mbps。加重压力并不会持续扩大优势。对高成本工况做吞吐恒等分解，绝大部分差额来自数据传输期间的有效速率。')
para('吞吐优先与保持可靠性优先会选出不同方案。在最紧的共享预算下，固定基线 F 比 RC 高 183.54 Mbps，但保持段几何失效达到 29.62%，RC 为 1.43%。若业务未指定可接受的失效率，单一吞吐排名不足以选定部署方案。')
sub('样本与材料')
table(['阶段','样本与用途'],[['复用离线数据','2,400 条单机路线，支持训练、验证、校准和历史测试'],['选型验证','20 个新验证世界，840 场网络运行，另有 3 组固定方案网格'],['独立测试','原计划 600 场，追加 60 场。共同覆盖 30 个测试世界，Chronos 子集为 10 个世界'],['完整交付','代码、模型、清单、原始指标、计算脚本、逐场与配对数据均可从报告所在目录进入']],[.24,.76])
para('阅读顺序：第 2–4 页介绍配置、机制与验证，第 5–8 页分别报告预算、成本、Chronos 和调度，第 9–10 页给出实现问题与数据复核入口。')

page('实验设计与度量口径')
para('一个物理世界同时确定轨迹、噪声和阴影。每种策略可独立决定何时观测，因而服务期间的观测历史会分叉。相同种子、相同外生世界使每个世界可作为自己的对照。路线类型覆盖稳定路径、全局或分段变速、停车、局部偏离后重入以及无可复用路径。默认 16 架中后者为 2 架。')
table(['要素','实际执行配置'],[
['链路与动作','28 GHz、400 MHz、32×32 阵列。两个轴分别取 8、16、32，共 9 个子阵组合。刷新期限为 0.5、1、2、3、5、8、10 s。'],
['时钟','180 s 预热后服务 180 s。真值步长 20 ms，预测每 0.5 s 一点，覆盖 10 s。TDMA 量子 10 ms，切换开销 0.1 ms。'],
['观测','一次全站独占 c 秒，其间停止发数；预算 ρ 约束全站长期感知占比。每次量测精度固定。仿真模型推理不占无线时钟。'],
['范围','固定 c=3.1 ms，ρ 为 5%、2.5%、5%/3。固定 ρ=5%，c 为 3.1、6.2、9.3、12.4 ms。基点复用。'],
['统计','各场先计算指标，按同一任务名配对，物理世界等权，有放回重采样 2,000 次，种子 271828，取 2.5% 与 97.5% 分位数。区间按比较分别计算。']],[.2,.8])
sub('一场实验算什么')
para('原计划 600 = 3 预算 × 4 方法 × 30 世界（360）+ 2 个额外成本 × 2 方法 × 30（120）+ 3 预算 × 2 个 Chronos 方法 × 前 10 世界（60）+ 2 个额外调度器 × 30（60）。追加 12.4 ms 时主实验已完成 538/600；沿用同一 30 个世界、冻结模型及 I/RC 两臂。660 表示配对运行次数。')
para('历史正式 FIVE 目录有 645 份网络指标，其中主比较 180 场，另有 MPC 120 场。本轮仅正式闭环网络运行即 840 + 600 + 60 = 1,500 场，固定网格、冒烟、单场诊断另计。两轮独立测试主线都保留 30 个世界；训练种子重复、消融等覆盖面有所不同。')
sub('如何读指标')
para('全网净吞吐是实际交付总比特除以 180 s。最差两机是在每个世界将 16 架按吞吐排序后取末两架平均。保持段按用户、预测包原点与波束划分，段内任一点几何损失超过 3 dB 即算失效；它与丢包率、数据时长占比各有分母。窄束占比按 32×32 波束占总用户保持时间计。覆盖率按不同的用户、预测包和选中期限计，先求世界内比例，再对世界求平均。')

page('修改后的代码平台与闭环流程')
table(['顺序','模块与运行行为'],[
['1 生成与预热','world.py 为各用户生成轨迹、观测噪声与阴影。预热按同一轮转协议产生真实观测，间隔受 max(1, Nc/ρ) 约束。服务开始时保留历史和预测包，重置执行令牌桶及 TDMA 轮次。'],
['2 发布中心','filters.py、memory.py、predictors.py 在新观测到达时更新。I 是位置交互多模型滤波预测，P 将历史观测片段按行进方向与弧长匹配并外推空间进度。RC 用门控权重融合 P 与 I 的中心。F 用 I 的中心及验证选定的固定束宽和周期。'],
['3 Chronos 对照','foundation.py 加载冻结的本地 Chronos-2 权重。C 读取因果历史；CR_raw 额外读取路线预测协变量。两者均使用自己的误差头进入同一闭环，测试抽查不含与 RC 同容量的 Chronos–IMM 门控融合。'],
['4 门控训练','horizon_gate.py 让上下文决定截距和正斜率：在发布时上下文固定，路线权重随 log(1+h) 单调。上下文包括近期 8 个观测间隔统计和历史专家误差。损失按各时域 IMM 残差尺度归一，再按旧回放的数据时间重加权。'],
['5 学习误差包络','按路线分组三折产生折外融合中心，重训 LightGBM 误差尺度头。使用 400 条独立校准路线、每条一个锚点，对每个候选期限的双轴全前缀最大归一化误差计算校准因子。五种预测中心使用同一种特征规格，各自拟合。'],
['6 分配预算','scheduler.py 对束宽和期限组合计算包络最坏损失及保守速率，经 3 dB 可行性筛选。value 按牺牲速率换取的负载下降修复动作，再在预算内改进目标。EDF 从各机最高速率候选出发；uniform 要求共用方形束宽。'],
['7 执行与评分','simulator.py 和 radio.py 用共享单次信用令牌桶执行感知，按最早到期处理请求。数据期按名义预测选 MCS，在真值、阴影下结算 BLER 与净吞吐。holding.py 事后评价保持段和选中期限覆盖。'],
['8 冻结与发布','ch6_matrix.py 生成任务，ch6_run.py 先验证再冻结、核验时钟与预算、原子发布。相同世界与预热可缓存。Chronos 用 GPU，其他臂主要用 CPU。追加队列沿用同一冻结模型与审计。']],[.23,.77])
para('置信度信息沿着“中心 → 双轴误差包络 → 可选束宽与期限 → 全站预算分配 → 实际有效速率”传递。因此闭环结果可以验证整套策略；单个模块的贡献另用同历史回放和验证消融检查。')

page('验证与冻结：哪些改动有数据支持')
para('验证世界为 chapter6/validation/1–20，已查看的开发世界 0 排除。840 场可拆为：旧模型下五个工作点及预热对照 360 场；三个端点上 RC 的门控、校准及 I/RC 的头规格对照 360 场；9.3 ms 处 P 和两种 Chronos 的 local/full 对照 80 场；紧预算 RC 的两种额外调度对照 40 场。两种 Chronos 在该验证比较各用前 10 个世界。')
sub('同世界、同闭环协议下的 RC 均值（净吞吐 Mbps，n=20）')
def vm(c,b,m,k):
    s=valid[(valid.method=='RC')&(valid.scheduler=='value')&(valid.warm=='lifecycle')&(valid.cost_ms==c)&np.isclose(valid.budget_pct,b)&(valid.model==m)&(valid.calibration==k)]
    return f(s.iloc[0].net_mbps)
table(['成本 / 预算','G0 全局','G1 全局','G0 按期限','G1 按期限','G1 全特征¹'],[[f'{c:g} / {budget(b)}%',*[vm(c,b,m,k) for m,k in [('G0','global'),('local','global'),('G0','deadline'),('local','deadline'),('full','deadline')]]] for c,b in [(3.1,5),(3.1,5/3),(9.3,5)]])
para('¹ 前四列共用 local 规格，最后一列为按期限校准的 full 头。9.3 ms、5% 时，仅改 G0 的校准增加 93.11 Mbps [85.26, 101.35]；固定全局校准，换 G1 及配套重训头增加 34.02 [27.34, 41.12]。在 G1 按期限下换 full 又增加 67.52 [56.19, 77.60]。不同起点的增量不宜直接相加。')
para('冻结规则只用 RC：三个端点各 20 个世界等权，全特征头均值 1038.93 Mbps，局部头 991.77。最终配置为 G1、全特征 LightGBM 头、按期限校准、value 调度。各方法共享头规格，但不同方法对规格偏好可能相反。9.3 ms 验证点的 I 从 local 的 743.7 降至 full 的 696.2 Mbps，因此测试中的 RC−I 也反映了这一选型规则。')
sub('预热、动作基线与离线诊断')
para('旧 G0/全局下，9.3 ms、5% 的固定低成本预热得到 I/RC 711.9/769.5 Mbps，全生命周期稀疏预热为 663.8/726.5。两种协议中 RC 都领先；预热配置影响绝对水平。新冻结模型正式测试仅覆盖全生命周期。F 在三个预算各用 20 个验证世界枚举 54、45、36 个可行动作；按吞吐择优均取 32×32，5%、2.5% 选 2 s，1.667% 选 3 s。')
para('旧 case0 同历史回放，0–1 s 角 RMSE 从 G0 的 0.21585° 降到 G1 的 0.18581°，同期 I 为 0.19008°；2–10 s 则由 0.19467° 增至 0.20631°。在原离线验证 400 条路线、1,200 个锚点上，G1/local 的 RC 窄束可发布 1,058 次，发布子集的选中期限覆盖 90.83%。短龄改善与长龄代价均应保留，这是离线及回放证据。')

page('共享预算主比较：吞吐、可靠性与弱用户')
para('以下均固定 3.1 ms、value 调度。每格为 30 个世界均值。F 的束宽和周期只在验证侧择优；它继承固定动作，表中 value 表示进入相同仿真入口。')
rows=[]
for b in [5,2.5,5/3]:
    for m in ['I','P','RC','F']:
        r=row_at(3.1,b,m);rows.append([budget(b),m,f(r.net_mbps),f(r.hold_fail_pct),f(r.worst10_mbps),f(r.sensing_pct),f(r.narrow_pct)])
table(['预算 %','方法','全网 Mbps','失效 %','最差两机 Mbps','感知 %','窄束 %'],rows,[.12,.09,.17,.14,.19,.14,.15])
sub('融合相对单一专家的增益随预算收紧而增大')
para('预算从 5% 降到 2.5%、1.667%，RC−I 从 +6.03 增至 +55.74、+234.03 Mbps，区间分别为 [5.38, 6.66]、[49.84, 61.93]、[223.93, 243.32]。RC−P 也为正，分别为 +5.61、+23.24、+37.51 Mbps。每档 RC 对 I、对 P 均在 30/30 个世界占优。最紧预算下，RC 实际感知 1.53%，与 I 接近，窄束保持占比却高 22.40 个百分点。')
para('最宽松点中 F 与 RC 的吞吐均值相差约 1 Mbps，RC−F 的区间 [−2.20, +0.01] 跨零。预算更紧时 F 的吞吐领先，同时其保持段失效从 6.74% 升到 29.62%。RC 在最紧预算相对 I 的失效也略高（1.43% 对 1.23%）。几何失效、低尾吞吐和全网吞吐应分别比较。')

page('固定 5% 成本压力线：12.4 ms 的边界')
fig1()
table(['单次成本','均匀间隔下界','I / RC Mbps','RC−I 与 95% 区间','胜出世界'],[[f'{r.cost_ms:g} ms',f'{16*r.cost_ms/50:.3f} s',f'{r.right:.2f} / {r.left:.2f}',ci(r),f'{r.wins}/30'] for r in pp.itertuples()],[.16,.18,.24,.29,.13])
para('9.3 → 12.4 ms 时，逐世界计算 (RC−I) 的变化为 −67.04 Mbps，95% 区间 [−95.13, −41.89]。I 的绝对吞吐从 715.37 降到 475.67，RC 从 945.60 降到 638.86。两者都受损，RC 的绝对降幅更大；12.4 ms 的相对百分比增益仍有 34.31%（两组均值之比），和以 Mbps 计的差值衡量不同。')
para('到 12.4 ms，I/RC 的窄束时间只剩 1.60%/9.07%，3 s 以上预测年龄占有效数据时间的 32.51%/34.22%；I/RC 各有平均 509/104 个规划帧请求负载超预算。RC 的实际保持失效更低（1.27% 对 2.80%，配对差 −1.54 个百分点，区间 [−2.52, −0.83]），最差两机仍高 4.69 Mbps。世界 22 中 RC 低约 16.8 Mbps。以上指标解释了高压下可选动作受限的现象；各环节单独造成多少回落，仍需干预对照。')
para('物理解释：c 是一次测量占用整个站点的时间，ρ=5% 意味着 180 s 服务期有约 9 s 感知时间，允许一次初始信用。12.4 ms 对应约 4.03 次全站观测/s，16 架均分约 3.968 s/架。协议允许一次感知跨越 10 ms 数据量子。这里固定量测精度，未采集真实设备端到端时延，9.3、12.4 ms 应作为高开销敏感性点。')

page('等间隔配对与 Chronos 子集')
sub('相同平均刷新机会，改变哪一个旋钮')
table(['Nc/ρ 下界','升成本 RC−I','降预算 RC−I','两种增益之差（Mbps，95% 区间）'],[['1.984 s','6.2 ms / 5%：51.62','3.1 ms / 2.5%：55.74','−4.12 [−7.10, −0.81]'],['2.976 s','9.3 ms / 5%：230.24','3.1 ms / 1.667%：234.03','−3.79 [−12.32, 4.97]']],[.14,.27,.29,.30])
para('两种控制产生相近的增益量级。1.984 s 的小差异区间排除零；2.976 s 的区间跨零，也没有预先指定等效界值。12.4 ms 对应的 3.1 ms、1.25% 等间隔对照未纳入本轮。')
sub('C 和 CR_raw 分别与 I、P、RC、F 配对')
para('下表给出 Chronos 方法减去各列方法的净吞吐（Mbps）。所有对照只取同一批 chapter6/test/0–9，不能直接用 Chronos 的 10 世界均值减去上一页的 30 世界均值。')
cr=pd.read_csv(primary/'chronos_paired.csv');z=cr[cr.metric=='net_mbps']
rows=[]
for b in sorted(z.budget_pct.unique(),reverse=True):
    for m in ['C','CR_raw']:
        zz=z[(z.budget_pct==b)&(z.method==m)].set_index('reference')
        rows.append([budget(b),m,*[f'{zz.loc[k,"difference"]:+.2f}' for k in ['I','P','RC','F']]])
table(['预算 %','方法','减 I','减 P','减 RC','减 F'],rows)
para('最紧预算下，C 与 CR_raw 超过 I，但仍落后 P、RC 和 F。CR_raw−RC 为 −136.86 Mbps [−152.39, −121.67]。在 5% 宽松点，此差值为 −14.90 [−17.79, −12.28]。三个预算中，原始 Chronos 加路线协变量比 C 的平均吞吐更高。两者与 RC 的排序仅覆盖这 10 个测试世界；完整的 24 个吞吐对照及其他指标区间见归档 chronos_paired.csv。')
para('9.3 ms 的 Chronos 结果来自验证侧，12.4 ms 的追加测试只包含 I 和 RC。独立压力线没有提供 Chronos 对 I/P/RC/F 的全成本扫描。')

page('调度归因与吞吐来源')
para('统一固定 RC、3.1 ms、1.667% 和 30 个世界，只改变调度规则。EDF 先选各机最高速率可行动作，再按到期先后竞争同一令牌桶；uniform 统一方形束宽并优先取长间隔；value 在速率与全站请求负载之间取舍。')
table(['调度','净吞吐 Mbps','保持失效 %','最差两机 Mbps','窄束 %','实际感知 %'],[[s,*[f(getattr(row_at(3.1,5/3,'RC',s),k)) for k in ['net_mbps','hold_fail_pct','worst10_mbps','narrow_pct','sensing_pct']]] for s in ['edf','value','uniform']],[.13,.19,.18,.21,.14,.15])
para('value 比 EDF 低 225.09 Mbps [−251.06, −202.04]，保持段失效低 15.09 个百分点 [−16.65, −13.56]。value 比 uniform 高 594.24 Mbps [568.69, 617.45]，同时失效高 0.74 个百分点。EDF 平均每场 969 个规划帧均请求超过预算；执行仍受令牌桶约束。可据业务容忍度选择操作点，本轮没有预设可靠性非劣界限。')
sub('RC−I 的 Mbps 来自多出的数据时间，还是数据期间传得更快？')
para('令 a 为数据秒数/180，v 为交付比特/数据秒，包含无可用预测时段的零收益。每个世界的净吞吐等于 a×v。以两方法均值作权重，对差值进行精确对称分解：时间项=(a_RC−a_I)×平均 v，速率项=(v_RC−v_I)×平均 a。下表先逐世界分解再求均值。')
dec=data('吞吐差对称分解').pivot(index=['cost_ms','budget_pct'],columns='term',values='delta')
table(['成本 / 预算','时间项 Mbps','有效速率项 Mbps'],[[f'{c:g} ms / {budget(b)}%',f'{r["数据时间项"]:+.2f}',f'{r["数据期有效速率项"]:+.2f}'] for (c,b),r in dec.iterrows()])
para('宽松点的 +6.03 Mbps 中，+4.72 来自数据时间；9.3 ms 下的 +230.24 中，速率项达 +230.36，时间项略负。这直接检验了“省感知时间带来优势”的假设。有效速率项同时受中心、束宽、MCS、可用性和实际调度影响，此分解本身不能把速率收益再逐一归因。')

page('第六章假设检验与实现边界')
table(['原报告问题','本轮证据与判断'],[
['6.4–6.5 短期路线权重过高','旧同历史回放支持 G1 短龄误差改善，长龄存在代价。20 世界验证消融支持 G1 配套新尺度头提升 RC 吞吐。首轮旧 RC−I 为 −18.02 Mbps，本轮新基点为 +6.03；世界、协议、门控和头同时改变，两数直接相减不能估计门控的因果效应。'],
['6.6 全局包络过于保守','期限校准在三个端点上均提高验证吞吐，紧预算及高成本增量尤大。测试 12.4 ms 的选中前缀覆盖为 RC 90.87%、I 89.76%。有限样本边际校准不会自动保证调度选中后的条件覆盖，F 的该指标为空。'],
['6.4 密集感知掩盖路线收益','两条压力主线均支持，12.4 ms 揭示绝对优势开始收窄。验证中的固定低成本预热对照支持服务期观测稀疏本身能产生影响。预热时长固定 180 s，本轮不能识别改变预热时长本身的因果效应。'],
['6.3 调度贡献与 6.7 动作评价','三臂区分了激进、统一加宽与价值调度的吞吐–失效取舍。F 和 EDF 对高吞吐的偏好支持继续研究显式可靠性约束与期望 goodput。期望速率动作评价、计划与执行统一 MCS 的在线闭环尚待单因素对照。'],
['6.8 MPC 与其他分支','MPC 刷新后状态近似修正和 2/4/6 s 窗口、Z 质量触发、同容量 CR-G、G3 及规模扩展均未纳入这 660 场。旧 MPC 价值误差可作独立背景，本报告的压力曲线不用于外推 MPC 效果。']],[.27,.73])
sub('代码审计发现的具体问题')
para('已经修正：原矩阵工况数与目标不符、缺 9.3 ms 和预算参数；旧汇总对预算分组不足；固定预热曾读入服务预算；F 赢家曾缺预算绑定并用到已查看的验证世界 0。当前清单、预热历史回归、F 绑定和服务时钟审计覆盖这些问题。归档的最近一次测试记录为 31 项通过。')
para('仍影响解释：候选速率按发布时前缀计算，其他用户刷新后会重选旧包的候选；规划用包络最坏损失，执行按名义 SNR 选 MCS。旧 case0 回放每 100 帧重评，RC 的 432 个抽查用户决策中有 67 个改变。原计划修复若耗尽可减载候选，value 会保留超载请求；高压场景已实际观察到。执行时间账本与请求可行性应分开读。')
para('证据边界：闭环只用训练种子 7；F 按验证吞吐择优，误差头按 RC 验证吞吐选型；多工况区间未作多重检验校正。后续若要声称可部署，宜先确定可接受的保持失效水平，用未参与选型的世界与额外训练种子复核，再以实测感知开销及在线推理时延校准本平台。')

page('代码与数据账本：从汇总数回到原始世界')
para('统一入口为 /data/chen/RouteClock_第六章完整实验_20261119。目录内前三项为通向同一 /data/chen 下既有档案的相对符号链接。原始批次、模型与清单保留发布时内容。')
table(['所在目录或文件','内容及正确用法'],[
['主实验归档/','指向 routeclock_archives/paired600。routeclock、tools、configs、tests 为代码和配置；checkpoints 为 Chronos-2 权重；results 为旧场回放、离线验证、G0/G1 模型及开发试验。'],
['主实验归档/runs/paired600/','manifest.json、独立测试600场.csv、验证任务.csv 为设计；models/frozen 为冻结模型和选择依据，fixed 为三套网格与赢家；validation、test 是各场原始产物。logs、probes、handover 与 dispatch_update 用于排查运行、缓存及代码版本。'],
['追加实验归档/','指向 routeclock_runs/paired600_pressure12_4。amendment.json 登记追加时点、模型和清单哈希；test 为 60 场原始数据。pressure_worlds 含四档 240 个方法×世界行，其中 180 行复用主实验。'],
['历史训练与基线/','指向 formal_execution/run_20261117_001732。five_data 的 train/validation/calibration/test 为 1200/400/400/400 个路线文件；five、mpc_formal 保留旧基线。requirements-resolved.txt 为运行环境依赖记录。'],
['数据表/ 与 Excel 工作簿','逐场指标 1,500 行、逐机资源 24,000 行；期限覆盖保留各 Δ 的分子分母，门控表按包数给出各时域均值。Excel 提供常用验证、测试、配对及分解视图。归档文件逐项索引记录真实路径、字节数、SHA-256。'],
['分析脚本/ 与 图/','汇总.py 从归档读 metrics 与 gate_weights 重新审计和计算，文件清单.py 重新验哈希，生成报告.py 从数据表生成本文和压力曲线。独立统计单位始终是世界。']],[.32,.68])
para('每场以 metrics.json 为数值来源，arguments.json 为运行参数，completed.json 为通过审计的标记。700 个 RC 场次另存 gate_weights.npz；缓存启用后的场次带 cache.json。正式矩阵使用 compact 模式，逐时隙 events、predictions、choices 仅在有相应日志的开发/回放任务可查；正式场的完整时序需按清单重放。')
para('完整性核验：主归档 SHA256SUMS 的 8,046 项及追加 sha256.json 的 278 项逐项复算一致。两个冻结模型 SHA-256 相同，始于 952eb45727a19741。历史清单的 argv 保留原 /data/coding 绝对路径；统计脚本映射到 /data/chen 后读取原件。复放时需映射脚本、模型、F 赢家和输出目录，使用独立输出目录避免重复发布。')
para('核对示例：在逐场指标中筛选 split=test、cost_ms=9.3、budget_pct=5、scheduler=value，以 task 对齐 I 和 RC。30 个差值均值为 +230.235 Mbps。metrics 列直接定位每个场次的源文件；原方案和追加的归属由 batch 列区分。')
fn='RouteClock_第六章完整实验分析报告'
doc.core_properties.title='共享感知预算下的 RouteClock：第六章完整实验分析报告'
doc.save(root/(fn+'.docx'))
def footer(canvas,document):
    canvas.setFont('CJK',8);canvas.setFillColor(colors.HexColor('#51656c'))
    canvas.drawString(58,26,'RouteClock  |  2026 年 11 月 19 日');canvas.drawRightString(535,26,str(document.page))
pdf=SimpleDocTemplate(str(root/(fn+'.pdf')),pagesize=A4,leftMargin=58,rightMargin=58,topMargin=43,bottomMargin=47,title=fn)
pdf.build(story,onFirstPage=footer,onLaterPages=footer)
