from ana import *
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer,Image
from xml.sax.saxutils import escape
font='/data/chen/teacher_report_20261118/assets/NotoSansSC-Regular.ttf'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
out=Path('/data/chen/refine-logs/RouteClock_0dB_结果分析报告_首版_20261123')
pages=[]
def page(title,sub):
 pages.append((title,sub,[]))
def add(k,v):pages[-1][2].append((k,v))
def p(s):add('p',s)
def h(s):add('h',s)
def note(s):add('note',s)
def tb(head,rows,width=None):add('t',(head,[[str(i) for i in r] for r in rows],width))
def f(x,n=2):return f'{x:.{n}f}'
def ci(v,n=2):
 return f'{v[0]:+.{n}f} [{v[1]:+.{n}f}, {v[2]:+.{n}f}]'
def vec(d,b,m,bm):
 return pair(d,b,m,bm)[1]
order=['F','I','RC','Chronos-2','VECTOR','RAFT']; mcolors=['#959a9f','#687496','#095f68','#d15a29','#5a8b49','#8866a3']
def fig_summary():
 fig,axes=plt.subplots(1,2,figsize=(8.3,2.9))
 for ax,tt,title in zip(axes,tab,['合成确认 · 16 机合计','实飞轨迹回放 · 单机']):
  for m,color in zip(order,mcolors):
   if m not in tt.index.get_level_values(1):continue
   ax.plot(range(3),[tt.loc[(b,m),'net_mbps'] for b in 'TML'],marker='o',ms=3.5,lw=1.6,color=color,label=m)
  ax.set_xticks(range(3),list('TML'));ax.grid(axis='y',alpha=.22);ax.set_title(title,fontsize=10);ax.set_ylim(800,1800);ax.tick_params(labelsize=8)
 axes[0].set_ylabel('净吞吐（Mbps）',fontsize=9)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=6,fontsize=8,frameon=False)
 fig.subplots_adjust(left=.10,right=.98,top=.85,bottom=.22,wspace=.22)
 path='/tmp/rc1123report/budgets.png';fig.savefig(path,dpi=210);plt.close(fig);return path
def mains(table,models):
 rows=[]
 for b in 'TML':
  for m in models:
   v=table.loc[(b,m)]
   rows.append([b,m,f(v.net_mbps,1),f(v.risk_pct),f(v.unavailable_pct),f(v.geom_pct),f(v.geom_given_action_pct)])
 tb(['预算','方法','G / Mbps','R / %','U / %','B / %','B | 动作 / %'],rows,[.09,.20,.16,.13,.13,.13,.16])
def contrasts_table(df,pairs,cluster=False):
 rows=[]
 for m,ref in pairs:
  for b in 'TML':
   dd,e=pair(df,b,m,ref,cluster=cluster)
   rows.append([f'{m} − {ref}',b,ci(e[0],1),ci(e[1]),f'{(dd.net_mbps>0).sum()}/{len(dd)}'])
 tb(['配对方向','预算','ΔG / Mbps [95% CI]','ΔR / pp [95% CI]','吞吐胜场'],rows,[.23,.07,.30,.28,.12])
synpar='合成 16 机，共享 T/M/L 感知额度 3/4.5/9 秒，每场服务 180 秒。先进模型采用 Chronos‑2 原生管、VECTOR 与 RAFT 各自的固定管；I、RC 各用归档误差头。自适应方法共用价值调度、到期刷新、令牌桶与轮转 TDMA，MCS 余量 0 dB。'
flypar='单机，每架次同趟预热 180 秒、计分 180 秒；T/M/L 感知额度依次为 0.1875/0.28125/0.5625 秒。I、RC 用各自归档默认管，Chronos‑2 用原生管，VECTOR、RAFT 用固定管；五法同一价值调度、到期刷新及 0 dB 无线模拟，先进模型固定种子 7。'
page('RouteClock 在有限感知预算下的表现','0 dB 完整实验结果分析报告 · 首版　｜　2026 年 11 月 23 日')
p('路线记忆的价值最清楚地体现在感知紧张时。与纯运动模型 I 相比，RC 在合成紧预算的 40 个世界、实飞紧预算的 59 个架次中，净吞吐全部提高。但两处的收益来源不同。')
note('本页 T 档均服务 180 秒。合成 16 机共享 3 秒感知额度，实飞单机为 0.1875 秒；I/RC 用各自默认管，Chronos‑2 用原生管，价值调度、到期刷新与 0 dB 余量一致。F 固定 8×32/3 秒。')
tb(['关键比较 · T 档','合成网络（40 世界）','实飞回放（59 架次）'],[
 ['RC − I 净吞吐','+229.0 Mbps（+22.6%）','+413.1 Mbps（+36.2%）'],
 ['RC − I 总风险','−0.94 个百分点','−24.71 个百分点'],
 ['收益的主要账本分量','可用时段内速率提高，贡献 +217.2 Mbps','可用发送时间增加，贡献 +406.7 Mbps'],
 ['Chronos-2 − RC 净吞吐','+171.9 Mbps，40/40 世界为正','+161.8 Mbps，58/59 架次为正']], [.27,.36,.37])
p('Chronos‑2 配原生误差管在两种环境的三个预算下均取得最高平均吞吐。RC 的优势应表述为相对 I 的预算效率提升。与固定周期 F 相比，RC 在合成 T 档提高吞吐，同时承担更高总风险；在 L 档才同时取得更高吞吐和更低风险。')
add('img',fig_summary())
note('图 1　横轴从紧到宽。合成图给出 16 机合计，实飞图给出单机模拟结果；跨图可比较排序和预算响应，绝对数值受站址、链路和共享服务条件影响。')
p('数据已于 11 月 23 日 02:41 完成汇总。此次逐场再核对了 2,505 个唯一任务键、公共计数、感知预算与净比特恒等式，并比对已发布总表。以下区间以世界或架次为单位。')
page('实验怎样把预测误差换算成通信结果','01　比较对象与计分合同')
p('基站每次付费观测后预测未来 10 秒的位置。调度器需要同时决定波束能收多窄、这次预测能用多久。中心轨迹决定指向，双角误差管给出可接受的偏差范围，两者共同决定可选动作。')
tb(['方法','中心预测与误差管','执行规则'],[
 ['I','交互多模型运动滤波（IMM）；I 专属归档 LightGBM 对数误差头及校准因子','自适应阵列与期限'],
 ['RC','已观测路线片段和沿线路程预测，按期限与 IMM 门控融合；RC 专属归档 LightGBM 误差头及校准','与 I 相同的调度器'],
 ['F','IMM 指向，固定 8×32 阵列、每 3 秒刷新；跳过误差管选动作','仍执行令牌桶、TDMA 与计分'],
 ['Chronos-2','全量微调；位置分位数经角度投影形成原生管，再作前缀校准','自适应，选型后冻结'],
 ['VECTOR / RAFT','速度 GRU 积分 / 多尺度历史片段检索；入选管各自由本中心的训练残差构成，按角轴与预测年龄变化','自适应，固定管仍可选不同束宽及期限']], [.19,.54,.27])
h('共同的物理和调度参数')
p('无线链路为 28 GHz、400 MHz、总发射功率 20 dBm；九种子阵列由两个轴在 8、16、32 中组合。每次感知占 3.1 ms，令牌桶至多容纳一次感知。预热后服务 180 秒，正向候选期限为 0.5、1、2、3、5、8、10 秒，预测网格间隔 0.5 秒。风险尺度为 1，最长持有 10 秒。')
p('先检查整个预测前缀的角度包络，保留最坏几何损失不超过 3 dB 的动作；按预测数据速率和感知开销取舍各机动作。最早到期者优先请求刷新，等令牌就绪；数据按 10 ms 时隙轮转，每时隙预留最多 0.1 ms 切换。动作过期后，相应分配时间交付为零。0 dB 是 MCS 余量，3 dB 是几何失准阈值。模拟计分未扣除模型的实测推理延迟。')
h('怎样读风险表')
p('每机每场 9,000 个公共计分点，间隔 20 ms。U 表示无可用动作占全部点的百分比，B 表示有动作且几何损失超过 3 dB 占全部点的百分比，总风险 R=U+B。B | 动作在每场的有动作点内计算，再等权平均。G 为成功交付比特除以完整 180 秒，单位 Mbps。差值中的风险单位是百分点（pp）。')
page('同一中心换误差管，收益取决于预算','02　开发选型 · 180 场')
p('参数　种子 7，10 个共同开发世界，16 机，T/M/L 全站额度 3/4.5/9 秒；每个中心两种候选管，三档均用共同价值调度、0 dB、尺度 1、最长 10 秒。同中心两管初始点预测哈希逐格一致；此后各臂按实际付费观测演化。')
p('三种中心均读取因果的 180 秒历史。合成训练路线 1,200 条中 960 条用于中心、240 条保留给误差估计；独立 400 条校准路线给出双角、全前缀的 90% 校准乘子，另有 400 条验证路线。外接 LightGBM 拟合 0.90 残差分位，固定管按角轴与预测年龄使用训练残差的 0.90 分位。原生方案在 Chronos‑2 用位置分位数，在另两模型用冻结表示上的误差头。')
rows=[]
for m in sb.MODELS:
 q=json.loads((root/'tubes'/m/'7/offline.json').read_text())
 for t in q['candidates']:
  vals=dev[(dev.model==m)&(dev.tube==t)].groupby('budget')[met].mean()
  rr=q['validation'][t];rows.append([m,t,*[f(vals.loc[b,'net_mbps'],1) for b in 'TML'],f(vals.net_mbps.mean(),1),f(100*min(rr['coverage']),1)])
tb(['中心','候选管','T 吞吐','M 吞吐','L 吞吐','等权吞吐','最低覆盖 %'],rows)
p('每中心先用验证覆盖与宽度筛出两管，再按三档等权平均 G 冻结一管，选型未设总风险门槛。最低覆盖为种子 7 验证锚点在七个正向期限中最低的前缀覆盖率；它与闭环 R 的分母和事件均不同。')
h('哪些选型结论稳，哪些依赖三档的权重')
p('Chronos‑2 原生管相对外接管，T/M/L 吞吐分别提高 240.7、203.2、77.9 Mbps。三档等权差 +173.9 Mbps，按开发世界配对的 95% 区间为 [156.9, 191.6]。T 档风险差 +0.09 pp，区间 [−0.06, +0.26]；该档支持吞吐收益，风险方向尚不清晰。')
p('VECTOR 和 RAFT 的固定管相对外接管，在 T 档分别少 74.7、65.0 Mbps，但风险分别低 3.37、3.43 pp；到 L 档反而多 173.2、123.6 Mbps。两者三档等权吞吐差仅 +9.7 [−10.9, 31.5] 和 +5.8 [−12.1, 22.1] Mbps。按规则应选固定管，但单独服务紧预算时，开发数据呈现另一种吞吐与风险取舍。')
note('开发区间为选型样本内的描述性配对重抽。同中心未入选管未在 40 个新世界并行复验，因此这些区间不承担独立确认的作用。')
page('合成确认：紧预算分开了三种策略','03　40 个新世界 · 1,440 场基线与模型闭环')
p('参数　'+synpar+' F 固定 8×32/3 秒。每世界各法共享物理轨迹、起始条件和时间噪声场；服务期内观测由各臂自行付费。先进模型先在每个世界等权平均种子 7/11/19，再平均 40 个世界。')
mains(tab[0],order)
p('T 档形成三种工作点：Chronos‑2 的 G 最高；RC 介于它和 F 之间；VECTOR、RAFT 的 G 均低于 900 Mbps，却有最低的几何失准贡献。若把 2% 作为描述性风险参考线，Chronos‑2、RC、I 的世界均值逐世界均高于该线，VECTOR、RAFT 的 40 个世界均在该线以内。吞吐排序不能代替风险约束下的选型。')
p('预算从 T 增至 M 时，VECTOR 与 RAFT 的平均风险反而升至 2.08% 和 2.41%，主要变化在 U。更宽的预算同时会改变刷新频率和调度器选取的期限；只看“预算更宽”不足以推出风险单调下降。')
page('RC 的收益随参照方法而改变','04　合成配对差值与不确定性')
p('参数　'+synpar+' F 固定 8×32/3 秒。下表在同世界做左减右，统一重抽 40 个世界 5,000 次，RNG 271828，报告双侧百分位 95% 区间。胜场按种子聚合后的吞吐严格大于参照计数。')
contrasts_table(syn,[('RC','I'),('RC','F'),('Chronos-2','RC')])
h('相对 I，重复路线信息的收益集中在预算紧张时')
p('RC 在 T/M 档对 I 的吞吐和总风险均有改善。L 档吞吐差缩至 +2.8 Mbps，区间跨零，风险反而高 0.11 pp。T 档 RC 的 U 比 I 低约 1.00 pp，但有动作条件下的失准率从 0.097% 升到 0.152%。总风险改善来自可用性，而非每次已选动作都更准。')
h('相对 F，紧预算下不能声称等风险双赢')
p('T 档 RC 比 F 多 94.9 Mbps，几何失准贡献由 1.390% 降至 0.146%，但 U 由 0.517% 升至约 3.56%，总风险净增 1.80 pp。L 档 RC 才同时带来 +415.0 Mbps 和 −0.76 pp。此处 F 的阵列和周期跨档固定，尚未做 F 与 RC 的同风险工作点搜索。')
p('Chronos‑2 相对 RC 的三档吞吐差均为正，且各为 40/40 胜场。T 档总风险虽低约 0.10 pp，绝对值仍达 3.60%。统计上可分辨的小差异与业务上能否接受该绝对风险，要分别判断。')
page('同样是吞吐改善，收益来自哪笔账？','05　从动作期限、有效窄束时间到交付比特')
p('参数　比较 T 档。合成 16 机共享 3 秒感知时间，实飞单机分摊 0.1875 秒；均计分 180 秒。两域均用相同价值调度与令牌桶，I/RC 用各自归档管，Chronos‑2 原生管，VECTOR/RAFT 固定管；合成 F 固定 8×32/3 秒。')
rows=[]
for domain,tt,models in [('合成',tab[0],order),('实飞',tab[1],['I','RC','Chronos-2','VECTOR','RAFT'])]:
 for m in models:
  r=tt.loc[('T',m)];rows.append([domain,m,f(r.obs,1),f(r.usable,2),f(r.active_rate,1),f(r.narrow,2),'固定 3' if m=='F' else f(r.hold)])
tb(['实验','方法','观测次数','有效发送 s','段内 Mbps','有效窄束 s','选定期限 s'],rows)
note('有效发送是服务时段扣除感知、切换和无动作分配后的时长；有效窄束统计使用 32×32 且几何损失 ≤3 dB 的发送时间。选定期限为刷新后确实选出动作时的期限均值，实际刷新间隔可能不同。上述量先逐场计算再取均值。')
h('用恒等式区分“多发了多久”和“发的时候有多快”')
p('设有效发送秒数为 S，段内比特率为 V，则 G=S×V/180。对每个配对，以两臂平均 V 乘 ΔS/180 计时间贡献，以平均 S 乘 ΔV/180 计速率贡献，两项恰好加总为 ΔG。该分解描述账本贡献，不单独证明某一模型模块的因果作用。')
tb(['紧预算配对','多出的有效时间贡献','段内速率贡献','合计 ΔG / Mbps'],[
 ['合成 RC − I','+11.8','+217.2','+229.0'],['实飞 RC − I','+406.7','+6.5','+413.1'],
 ['合成 Chronos-2 − RC','+2.9','+169.0','+171.9'],['实飞 Chronos-2 − RC','+103.5','+58.3','+161.8']])
p('合成 RC 与 I 的刷新次数接近，RC 的有效窄束时间从 16.47 增至 40.47 秒，足以解释为何段内速率成为主要差异。实飞 T 档两者均约刷新 60 次，段内速率近似，但 RC 多获得 44.10 秒有效发送时间。相同的吞吐方向来自两种不同的预算瓶颈。')
page('预算放宽后，哪些差异仍然存在？','06　合成跨种子与服务尾部')
p('参数　'+synpar+' 下表每行是一份冻结中心与该种子的校准包，在每档 40 个相同世界中的均值。三种子共用开发时冻结的误差管类型。')
rows=[]
for m in sb.MODELS:
 for s in (7,11,19):
  group=syn[(syn.model==m)&(syn.seed==s)].groupby('budget')[met].mean()
  rows.append([m,s,*[f(group.loc[b,'net_mbps'],1) for b in 'TML'],*[f(group.loc[b,'risk_pct']) for b in 'TML']])
tb(['中心','种子','G · T','G · M','G · L','R% · T','R% · M','R% · L'],rows)
p('Chronos‑2 三个种子的 T 档均值介于 1,411.6–1,421.0 Mbps，均高于 RC 的 1,244.3。RAFT 的 M 档差异最大：种子 11 达 1,160.0 Mbps，种子 19 为 1,014.4；同期 U 与几何失准合计的 R 从 1.77% 升至 3.34%。种子 19 的有效时段内速率仅 1,084.6 Mbps，种子 11 为 1,220.9。中心和校准头均随种子改变，现有对照无法再把差异分摊到二者。')
h('低风险与高服务水平并非同一个指标')
p('合成 T 档，VECTOR 与 RAFT 的净吞吐接近 895 Mbps、有效 32×32 窄束时间接近零，均值选定期限约 3 秒。到 L 档，两者有效窄束时间分别达到 167.97 和 168.36 秒，选定期限均接近 1 秒，吞吐均超过 1,570 Mbps。预算充足时可更频繁刷新，从而在有效时间内支持更高段内速率；这个变化同时解释了固定管在 L 档开发选型中的收益。')
p('按每场 16 架机各自交付比特折算用户吞吐，再取机间第 10 百分位，T 档均值分别为 I 54.5、F 64.2、RC 67.0、Chronos‑2 77.7、VECTOR 49.2、RAFT 49.2 Mbps。RC 相对 I 的改善在低吞吐用户处也可见；该结果是服务尾部描述，尚未单独检验公平性。')
note('训练验证的七期限双角误差约为 Chronos‑2 0.0131、RAFT 0.0238、VECTOR 0.0310 rad。RAFT 开环优于 VECTOR，但两者合成 T 档吞吐几乎相同；点误差排序无法代替“误差管—可行动作—实际交付”的闭环测量。')
page('实飞轨迹怎样进入无线回放？','07　AMOVFLY 窗口、预算与结果全表')
p('270 架次经连续性、坐标和离地高度筛选，127 个质量候选中排除 65 个非规则环线和 3 个连续周期片段不足的架次，最终纳入三角环线 R1 共 20 架次、矩形环线 R2 共 39 架次。每架次取一个连续窗口，原始间隔 ≤0.25 秒，离地至少比起始首秒中位数高 5 米；预热和计分各 180 秒，另留 10.02 秒真值。冻结范围未设最低圈数。')
p('参数　'+flypar+' 每次感知 3.1 ms，三档分别相当于全时段的 0.1042%、0.15625%、0.3125%；单机预热额度 0.05/16，约每秒取得一次观测。实测经纬度投影至共用场地坐标系，虚拟站位于场地参考点以西 100 米、高 5 米。真实轨迹与速度用于回放，观测噪声、阴影、发包和净吞吐按模拟器生成。')
mains(tab[1],order[1:])
p('Chronos‑2 的平均吞吐从 T 到 L 仅增加 0.83 Mbps；I 同期增加 571.19 Mbps，RC 增加 159.37 Mbps。L 档五种方法的均值只相差约 6.2 Mbps。28 GHz/400 MHz 模拟链路的最高名义有效速率约 1,739.2 Mbps（400×0.85×5.1152），还要扣除切换、感知和误包；在该几何条件下靠近速率上限，会压缩方法间的吞吐差距。')
page('实飞外验：平均优势覆盖了多少架次？','08　架次配对及按飞行日期聚类的敏感性')
p('参数　'+flypar+' 每个主对比含 59 对相同架次。下表的置信区间按架次重抽，ΔR 以百分点计。F 未进入实飞清单。')
contrasts_table(fly,[('RC','I'),('Chronos-2','RC'),('VECTOR','RC'),('RAFT','RC')])
p('T 档 RC 的总风险从 I 的 31.18% 降至 6.48%，主要由无动作占比从 30.96% 降至 6.17% 驱动。有动作条件下的几何失准率分别为 0.38% 和 0.35%。Chronos‑2 的 U 为零，使用约 31 次观测就得到 178.10 秒有效发送时间；它的条件失准率 0.50% 反而高于 RC。这解释了“总风险更低”与“出束后的指向未更准”为何能同时成立。')
p('59 架次来自 13 个飞行日期，单日架次数为 1–13。整日重抽、保留日内各架次后，T 档 RC − I 的 ΔG 区间为 [363.3, 453.4] Mbps、ΔR 为 [−27.74, −21.22] pp；Chronos‑2 − RC 分别为 [108.6, 232.1] 和 [−9.55, −3.23]。两项紧预算结论对这一聚类口径稳定。')
p('在 L 档，RC − I 的平均增益仅 +1.3 Mbps、22/59 架次为正，按日期重抽的区间 [−0.3, +3.6] 跨零。Chronos‑2 − RC 的吞吐多 3.3 Mbps，风险差区间跨零。绝对差异和胜场数回答不同问题，主文宜同时报告。')
page('三角与矩形环线揭示了不同的失效位置','09　紧预算实飞子组 · 探索性分析')
p('参数　T 档，每架次 0.1875 秒服务感知额度、观测成本 3.1 ms。每架次同趟预热 180 秒，五法共享预热观测与同一时间噪声；单机价值调度、0 dB。I/RC 为各自默认管，Chronos‑2 原生管，另两模型固定管；先进模型仅种子 7。')
rows=[]
for route,grp in fly[fly.budget=='T'].groupby('route'):
 for m in ['I','RC','Chronos-2','VECTOR','RAFT']:
  vals=grp[grp.model==m][met].mean();rows.append([route.split('_')[-1],m,f(vals.net_mbps,1),f(vals.risk_pct),f(vals.unavailable_pct),f(vals.geom_pct)])
tb(['航线','方法','G / Mbps','R / %','U / %','B / %'],rows)
p('R1 三角航线，RC 比 I 多 460.1 Mbps，但 U 仍有 15.77%。Chronos‑2 在全部 20 架次超过 RC，平均再多 349.2 Mbps [299.5, 395.3]。R2 矩形航线，RC 已将 U 降至 1.25%，Chronos‑2 的平均增益缩至 65.7 Mbps [46.2, 87.0]。两组中 Chronos‑2 的绝对吞吐几乎相同，子组差异主要体现在 RC 留下多少空档。')
p('按架次计算，实飞 T 档净吞吐的第 10 百分位为 I 808.7、RC 1,309.8、Chronos‑2 1,714.0 Mbps。RC 的 59 架次均比 I 吞吐高，但只有 31 架次的总风险 ≤2%；Chronos‑2、VECTOR、RAFT 各有 53 架次达此参考线。均值风险低于 2% 无法保证每趟均满足要求。')
p('预热/计分段的等效圈数中位数在 R1 为 6.02/6.03，在 R2 为 5.96/5.85。两组均有重复运动，单凭圈数无法解释差距；航线、机型、速度、日期和站机几何未随机化。转弯附近的门控与动作过期是可检验的解释，当前逐场汇总无法定位损失具体发生在哪个弯。')
page('哪些结论可以写入论文主线？','10　证据解释、边界与可复核来源')
h('主结论应限定参照、预算和环境')
p('① 与纯运动基线 I 比较，路线辅助系统 RC 在两种环境的 T/M 档均提高吞吐并降低总风险。紧预算收益在全部 40 个合成世界和全部 59 个实飞架次出现。L 档的吞吐增益区间跨零。研究贡献可以落在“有限感知下，路线辅助延长可用服务或提高段内交付效率”。')
p('② RC 与固定 F 的关系取决于风险口径。T 档 RC 出束后的指向失准更少，却更常没有可用动作。用总风险考察，RC 在 L 档取得双赢，在 T/M 档体现风险与吞吐的交换。只引用 B 或只报告吞吐都会遗漏这一结果。')
p('③ 强预测模型是必要参照。Chronos‑2 的原生分位管在开发集有清楚的同中心收益；冻结后组合在六个场景×预算工作点均有最高平均 G。VECTOR、RAFT 的固定管更保守，在合成 T 档和实飞 T 档呈现不同的相对排名。可以主张误差管与可用动作必须一起评估。')
h('还缺哪种证据')
p('归因到“路线记忆本身”，需要在相同中心、误差头和预热快照下做保留/清空记忆对照。RC − I 同时涉及路线预测、门控和专属误差头。归因到同风险吞吐优势，需要先在开发世界匹配总风险再独立确认；本轮按吞吐选管。若解释三角航线失效，优先核对刷新前后的路线支持、剩余有效期限及令牌等待与转弯时刻。以上是后续验证方向，本报告未将其计入 2,505 场。')
p('本轮使用单一 0 dB MCS 余量、一个实飞场地假定站址，实飞仅种子 7、单机、规则环线。实飞阶段验证轨迹迁移下的模拟链路表现；多机同站实测、跨架次记忆、真实射频吞吐和实际推理时延仍需独立证据。子组、账本分解及 2% 参考线属于补充描述；配对区间为逐比较区间，未作多重比较校正。')
h('数据与复算索引')
p('工作根目录 /data/chen/routeclock_runs/confirm_0db/。冻结范围见 scope.json；实际执行见 run_baselines.py、run_advanced.py、run_flights.py。模型权重记录位于 training/，误差管、验证覆盖和选型均值位于 tubes/，实飞窗口与排除账本位于 work/flight_windows.json、work/flight_exclusions.csv。')
p('results/summary/all_cells.csv 可复算各章 G、U、B、R；paired_effects.csv 提供先进模型对基线的配对区间。RC − I、RC − F、开发管间对照、子组区间按同一 5,000 次世界/架次配对重抽补算。机制章节直接读取逐场 complete.json 的 ledger_s、bits、updates、selected_deadlines 与 narrow_good_data_s。')
page('先进模型与传统策略的其余合成对照','附表　各预算的配对吞吐与总风险')
p('参数　'+synpar+' F 固定 8×32/3 秒。独立单位为 40 个新世界，模型的三个种子先在世界内平均。每格为左减右的配对均值及双侧 95% 区间。')
contrasts_table(syn,[('VECTOR','RC'),('RAFT','RC'),('Chronos-2','F'),('VECTOR','F'),('RAFT','F')])
p('合成紧预算下，VECTOR、RAFT 比 RC 吞吐偏低而总风险更低；实飞紧预算下两者又比 RC 多约 96 Mbps。合成中 16 机竞争感知资源和数据时隙，实飞回放中单机使用分摊感知额度、拥有自己的数据时隙，场地几何也不同。这个排序反转说明两个验证场景都值得保留；仅凭现有设计不能将反转独立归因于机数、航线形状或训练域。')
note('其余先进模型相对 I 的逐项区间，以及无动作、几何失准、条件几何失准的完整配对差值见同批 paired_effects.csv。显示值均经过四舍五入，表内分量相加可能与总量相差末位。')
# Editable document and fixed-layout reading copy share the same page blocks.
pdfmetrics.registerFont(TTFont('RCFont',font))
paper=A4; W,H=paper
def sty(name,size,lead,space,color='#172d37'):
 return ParagraphStyle(name,fontName='RCFont',fontSize=size,leading=lead,spaceAfter=space,textColor=colors.HexColor(color),wordWrap='CJK',splitLongWords=1)
styles={k:sty(k,*v) for k,v in {'title':(22,29,9,'#104854'),'sub':(10,15,4,'#557077'),'p':(11.2,17.8,8,'#182f37'),'h':(12,18,4,'#08645f'),'note':(9.3,14,6,'#486168'),'cell':(9.4,13.5,0,'#19333b'),'th':(9.4,13.5,0,'#ffffff')}.items()}
def para(s,k):return Paragraph(escape(s),styles[k])
pdf=SimpleDocTemplate(str(out)+'.pdf',pagesize=A4,leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=44,title='RouteClock 0 dB 完整实验结果分析报告',author='RouteClock')
story=[]
doc=Document();section=doc.sections[0];section.page_width=Cm(21);section.page_height=Cm(29.7);section.top_margin=Cm(1.6);section.bottom_margin=Cm(1.6);section.left_margin=Cm(1.7);section.right_margin=Cm(1.7)
def drun(run,sz=10,hex='172D37',bold=False):
 run.font.name='Noto Sans SC';run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑');run.font.size=Pt(sz);run.font.color.rgb=RGBColor.from_string(hex);run.bold=bold
def dp(s,size=10,color='172D37',after=5,bold=False,typ=None):
 v=doc.add_paragraph(style=typ);v.paragraph_format.space_before=Pt(0);v.paragraph_format.space_after=Pt(after);v.paragraph_format.line_spacing=1.16;drun(v.add_run(s),size,color,bold);return v
foot=section.footer.paragraphs[0]; drun(foot.add_run('RouteClock   /   0 dB 完整实验　　2026.11.23　　　　　　　　　'),8,'577279');fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)
for index,(title,sub,blocks) in enumerate(pages):
 if index:story.append(PageBreak());doc.add_page_break()
 story.extend([para(sub,'sub'),para(title,'title')]);dp(sub,9,'527579',4);dp(title,20,'104854',9,True,'Heading 1')
 for k,val in blocks:
  if k in ('p','h','note'):
   story.append(para(val,k));dp(val,{'p':10,'h':12,'note':8.5}[k],{'p':'172D37','h':'08645F','note':'486168'}[k],5,k=='h');continue
  if k=='img':
   story.extend([Image(val,width=490,height=171),Spacer(1,5)]);doc.add_picture(val,width=Cm(17));continue
  hd,rows,wt=val; wt=wt or [1/len(hd)]*len(hd)
  data=[[para(v,'th' if i==0 else 'cell') for v in row] for i,row in enumerate([hd]+rows)]
  tbl=Table(data,colWidths=[(pdf.width-12)*x for x in wt],repeatRows=1,hAlign='LEFT')
  tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#19565d')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf3f2')]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#9eafb1')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
  story.extend([tbl,Spacer(1,7)])
  dt=doc.add_table(rows=0,cols=len(hd));dt.autofit=False
  for j,w in enumerate(wt):dt.columns[j].width=Cm(17.6*w)
  for i,row in enumerate([hd]+rows):
   cells=dt.add_row().cells
   for j,v in enumerate(row):
    cell=cells[j];cell.width=Cm(17.6*wt[j]);pp=cell.paragraphs[0];pp.paragraph_format.space_after=Pt(2);pp.paragraph_format.space_before=Pt(2);pp.paragraph_format.line_spacing=1.05
    drun(pp.add_run(v),8.5,'FFFFFF' if i==0 else '172D37',i==0)
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'19565D' if not i else ('EDF3F2' if i%2==0 else 'FFFFFF'));cell._tc.get_or_add_tcPr().append(sh)
def footer(c,d):
 c.saveState();c.setStrokeColor(colors.HexColor('#c7d3d4'));c.line(42,34,W-42,34);c.setFont('RCFont',8);c.setFillColor(colors.HexColor('#536d72'));c.drawString(42,21,'RouteClock　|　0 dB 完整实验　·　首版');c.drawRightString(W-42,21,f'2026.11.23　 /　{d.page}');c.restoreState()
pdf.build(story,onFirstPage=footer,onLaterPages=footer)
doc.core_properties.title='RouteClock 在有限感知预算下的表现';doc.save(str(out)+'.docx')
import fitz
with fitz.open(str(out)+'.pdf') as result:
 for i,pg in enumerate(result): print(i+1,pg.get_text().splitlines()[2:4])
 print('pages',len(result),'expected',len(pages))
 assert len(result)==len(pages)
