"""Build the Chinese report from verified paired receipts and archive tables."""
from pathlib import Path
import json
import statistics as st
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

O=Path(__file__).resolve().parent;root=Path('/data/chen')
D=pd.read_csv(O/'闭环配置均值.csv')
A=pd.read_csv(O/'同历史输入与训练.csv').groupby(['model','horizon_s']).mean(numeric_only=True)
H=pd.read_csv(O/'门控与专家上限.csv').groupby(['model','horizon_s']).mean(numeric_only=True)
F=pd.read_csv(O/'配对因素效应.csv').set_index('比较')
exact=pd.DataFrame([json.loads(f.read_text()) for f in (O/'exact_old').glob('*.json')]).groupby('method').mean(numeric_only=True)
M=pd.DataFrame([json.loads(f.read_text()) for f in (O.parent/'MCS余量配对').glob('*.json')]).groupby(['model','mcs_margin_db']).mean(numeric_only=True)
def row(name,tube=None,scene='new',warm='dense',policy='strict',margin=0):
    if tube is None:tube='default' if name in ('RC','P','I','C19') else 'native'
    s=D[(D.model==name)&(D.tube==tube)&(D.scene==scene)&(D.warm==warm)&(D.policy==policy)&(D.margin==margin)]
    assert len(s)==1
    return s.iloc[0]
def gap(**kw):return row('tuned',**kw).G-row('RC',**kw).G
def fmt(v):return f'{v:.2f}'
def sg(v):return f'{v:+.2f}'
font=root/'teacher_report_20261118/assets/NotoSansSC-Regular.ttf'
pdfmetrics.registerFont(TTFont('NotoC',str(font)))
def style(name,size,leading,color='#233545',after=6):
    return ParagraphStyle(name,fontName='NotoC',fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=after,wordWrap='CJK',alignment=TA_LEFT)
styles={'title':style('title',21,31,'#183e54',12),'sub':style('sub',12,18,'#275970',6),'p':style('p',9.7,15.7),'small':style('small',8.3,13,'#445968'),'cell':style('cell',8.35,12,after=0),'th':style('th',8.6,12,'#ffffff',0)}
s=[]
def para(t,k='p'):s.append(Paragraph(escape(t),styles[k]))
def table(data,widths):
    rows=[[Paragraph(escape(str(v)),styles['th' if r==0 else 'cell']) for v in x] for r,x in enumerate(data)]
    tb=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#245268')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    s.extend([tb,Spacer(1,9)])
def new(t):s.extend([PageBreak()]);para(t,'title')
def footer(canvas,doc):
    canvas.setTitle('RouteClock · RC 与 Chronos-2 跨版本归因')
    canvas.setFont('NotoC',8);canvas.setFillColor(colors.HexColor('#496474'));canvas.drawString(40,27,'RouteClock  |  合成闭环与固定历史诊断  |  2026-11-23')
    canvas.drawRightString(555,27,f'{doc.page}')
para('RC 与 Chronos-2 的排名为什么反转','title')
para('11 月 19 日 GitHub 第六章固定历史修订版，与 11 月 23 日 0 dB 确认版的配对归因','sub')
para('历史成绩先在各自批次内配对。旧批次只纳入 Chronos 跑过的 10 个世界；本轮在每个世界内平均三个训练种子，再比较 40 个世界。表中差值均为 Chronos 减 RC，单位 Mbps。')
table([['服务期共享感知预算','11 月 19 日差值［95% 区间］','本轮差值［95% 区间］'],['紧档，1.667%，约 3 秒','−66.88［−78.36，−52.46］','+171.92［+159.58，+185.11］'],['中档，2.5%，4.5 秒','−88.68［−100.67，−77.39］','+101.18［+94.18，+108.43］'],['松档，5%，9 秒','−19.95［−24.70，−15.35］','+35.51［+31.76，+39.25］']],[155,180,180])
para(f'锁定 11 月 19 日双方原装中心、误差头、5 dB 和四个旧世界的同一预热状态，单换整个执行平台，C−RC 从 {sg(exact.loc["C","archive_G"]-exact.loc["RC","archive_G"])} 变为 {sg(exact.loc["C","modern_G"]-exact.loc["RC","modern_G"])} Mbps。再用本轮完整系统（0 dB、微调 Chronos、原生管、归档 RC 专用头、当期预热规则）跑这四个旧世界，差值为 {sg(gap(scene="old"))} Mbps。这三步分别显示两套平台的工作点变化，以及本轮完整预测系统在旧世界上的表现。')
para('相同世界、相同资源下，单独改变关键因素','sub')
para(f'在紧档四世界中，固定新输入和原生管流程，仅把 Chronos 从预训练权重换为微调权重并为中心重校准，吞吐改变 {sg(row("tuned").G-row("frozen").G)} Mbps。固定微调权重，把专用分位树换成原生管，改变 {sg(row("tuned").G-row("tuned","lightgbm").G)} Mbps。这两项有交互，不能相加作为对历史翻转的唯一分摊。')
para(f'本轮两法都使用外接分位树、同规格特征和同一训练校准规则时，微调 Chronos 减 RC 为 {sg(row("tuned","lightgbm").G-row("RC","lightgbm").G)} Mbps；两法各用已发布误差管时为 {sg(gap())} Mbps。退回 5 dB，四世界微调 Chronos 仍领先 214.89 Mbps。')
para(f'旧冻结 C 连同原误差头，5 dB 下在旧四世界落后 {abs(row("C19",scene="old",margin=5).G-row("RC",scene="old",margin=5).G):.2f} Mbps，在新四世界却领先 {row("C19",margin=5).G-M.loc[("RC",5),"G_mbps"]:.2f} Mbps。换样本足以改变接近持平的旧式方案的名次；新输入与微调管的实验则解释为何当前领先幅度更大。')
para('参数与样本','sub')
para('除显式对照外，使用 16 机共享、180 秒密集预热（5%）及 180 秒服务，单次观测 3.1 ms，28 GHz / 400 MHz，value 调度、9 种阵列、0.5 至 10 秒的 7 档期限、3 dB 几何门限，0 dB MCS 余量。基础 RC 为归档 G1 路线记忆＋IMM 中心及其专用误差头；基础 Chronos 为种子 7 的全参数微调中心及原生分位管。')
para('专项复用此前按系统抽取的新确认世界 0、10、20、30，旧世界选 0、2、5、7；共 120 场：96 场因素试验、16 场 MCS 配对复跑、8 场读取原始起点并切换执行器。另有两套各 40 世界 × 16 机的同历史诊断。四世界效应报告均值、范围和方向，不能作为全体确认世界的精确效应。','small')
new('训练、输入和误差管')
para('各格复用同一批四个新世界、同一紧档闭环。每行各自校准原生管和外接分位树。外接分位树统一采用 0.90 分位损失、15 叶、130 棵树，同 240 条训练路线拟合，同 400 条路线做双角前缀校准。原生管由位置分位投影。')
rr=[['预测中心','原生管 G / 风险','外接分位树 G / 风险','原生减树 Mbps']]
for name,cn in [('old','官方冻结，旧输入'),('frozen','官方冻结，新输入'),('tuned','全量微调，新输入')]:
    n,t=row(name),row(name,'lightgbm');rr.append([cn,f'{n.G:.2f} / {n.risk_pct:.2f}%',f'{t.G:.2f} / {t.risk_pct:.2f}%',sg(n.G-t.G)])
table(rr,[144,138,139,94])
para(f'固定新输入，微调在外接树下改变 {sg(row("tuned","lightgbm").G-row("frozen","lightgbm").G)} Mbps，原生管下改变 {sg(row("tuned").G-row("frozen").G)} Mbps。交互量为二者之差。原生管跟随检查点输出分位宽度；外接树也要用新中心的残差重拟合。这些格度量的是可部署系统的增益。中心本身的变化见下一页。')
para('把旧版接到同一个本轮平台','sub')
g0=row('RC').G
steps=[('旧输入冻结权重 + 11 月 19 日原装 C 头与乘子',row('C19').G),('同一冻结中心 + 新规格外接分位树',row('old','lightgbm').G),('新输入冻结中心 + 同规格外接分位树',row('frozen','lightgbm').G),('新输入微调中心 + 同规格外接分位树',row('tuned','lightgbm').G),('同一微调中心 + 原生分位管',row('tuned').G)]
rr=[['新平台上的逐步替换','Chronos G','减基础 RC','相对上一步']]
for j,(label,value) in enumerate(steps):rr.append([label,fmt(value),sg(value-g0),sg(value-steps[j-1][1]) if j else '—'])
table(rr,[266,75,83,91])
para(f'基础 RC 为 {g0:.2f} Mbps；改用与 Chronos 完全相同训练协议的专用分位树后，RC 为 {row("RC","lightgbm").G:.2f} Mbps。两种外接头拥有相同规格时，各自仍随中心残差校准。上表增量依赖替换次序；相同输入、纯中心预测误差才可直接比较中心精度。','small')
new('固定同一观测，分开检查输入和权重')
para('40 个新世界，各 16 架。预热密集轮转，随后固定紧档均匀轮询，每架间隔 2.976 秒，到任务第 300 秒前最后一次观测作锚点。共 640 份完整因果历史，所有中心使用同一实测，独立于各自的闭环调度。32×32 可保持比例按真值对整段 5 秒检查几何损失 ≤3 dB。')
rr=[['输入及权重','5 秒位置误差 m','5 秒双角误差 °','5 秒 32×32 可保持','10 秒双角误差 °']]
for key,cn in [('old','冻结，旧接口，720 格'),('old360','冻结，旧接口，360 格'),('old360nopast','冻结，360 格，移除协变量'),('frozen','冻结，新接口，360 格'),('tuned','微调，新接口，360 格')]:
    v=A.loc[key,5];rr.append([cn,fmt(v.position_m),fmt(v.angle_deg),f'{v.safe32*100:.1f}%',fmt(A.loc[(key,10),'angle_deg'])])
table(rr,[151,83,87,108,86])
para('旧接口将最近实测向前填充，另输入量测掩码、年龄、径向速度和测量不确定度等。去掉协变量的旧接口仍保留位置填充。新接口按当前观测中心化三轴位置、仅保留半秒内有实测的格点，并显式传掩码；旧无协变量接口到新接口的变化包含缺测表示和推理调用方式。')
para('单纯缩短历史和更换输入没有重现微调后的精度。固定新输入后，冻结与微调的每个锚点、预测时域、解码接口完全一致，权重变化对角误差的影响可以直接解释。训练时以三轴未来位置的分位预测损失更新全模型参数（AdamW，学习率 1e−6，批量 8，最多 12 轮），用多期限双角误差选择检查点；通信吞吐只在开发集选择部署用管时使用。')
para('用正式确认结果检验通信路径','sub')
para('先前的 40 世界、三种子正式紧档分解显示，Chronos 对 RC 的 +171.92 Mbps 中约 +169.0 来自有效时段内速率，约 +2.9 来自有效时间变化。有效 32×32 发送时长由 40.47 增至 88.53 秒。共同预热末锚点处，5 秒实际可保持 32×32 的比例为 RC 73.1%、微调 Chronos 85.0%，两种专用管放行为 30.0% 和 56.9%。不同锚点和时钟的数字应分别解读。')
new('预热、余量、动作规则与世界批次')
para('除最后两行外均比较同一批四个新世界。吞吐按完整 180 秒归一，差值方向为微调 Chronos 减基础 RC。两法的中心、管、紧档服务预算和 3 dB 几何门限均锁定。')
rr=[['条件','Chronos Mbps','RC Mbps','差值 Mbps']]
def env(label,a,b):rr.append([label,fmt(a),fmt(b),sg(a-b)])
env('密集预热，严格到期，0 dB',row('tuned').G,row('RC').G)
env('改稀疏预热，服务预算相同',row('tuned',warm='sparse').G,row('RC',warm='sparse').G)
env('改 5 dB MCS 余量，两臂重新闭环',M.loc[('Chronos-2',5),'G_mbps'],M.loc[('RC',5),'G_mbps'])
env('相同已发动作，到预测包 10 秒才停止记账',row('tuned').ten_s_G,row('RC').ten_s_G)
env('允许重选过期候选，仍按所选期限停止交付',row('tuned',policy='reuse').G,row('RC',policy='reuse').G)
env('同时放宽候选重选与交付有效期',row('tuned',policy='reuse').ten_s_G,row('RC',policy='reuse').ten_s_G)
env('规划收益估计按保守 SNR 选档',row('tuned',policy='conservative').G,row('RC',policy='conservative').G)
env('改用 11 月 19 日四个物理世界，本轮整套系统',row('tuned',scene='old').G,row('RC',scene='old').G)
env('同一批旧世界，移植旧 C 头与中心，5 dB',row('C19',scene='old',margin=5).G,row('RC',scene='old',margin=5).G)
table(rr,[266,87,80,82])
para(f'用于查明旧模型如何改变排名的交叉对照：保持第六章冻结 C 的原装头和当期 RC，本轮平台在旧四世界 5 dB 的差值为 {sg(row("C19",scene="old",margin=5).G-row("RC",scene="old",margin=5).G)}，在新四世界 5 dB 为 {sg(row("C19",margin=5).G-M.loc[("RC",5),"G_mbps"])}，在同一新四世界 0 dB 为 {sg(row("C19").G-M.loc[("RC",0),"G_mbps"])} Mbps。前一项比较是换样本的描述，后一项才是同世界的余量效应。','small')
para('余量同时影响 MCS 门限和误块率曲线。四世界内退回 5 dB 并未恢复旧排名。几何门限用于阵列准入；改变规划用 SNR 时，保持同一覆盖包络和射频结算，只改调度对候选动作的收益打分。')
para('过期候选试验锁定当前请求时刻及前沿生成规则，只改选动作时能否再次使用过期候选；“10 秒才停止记账”对同一动作日志重新计分，已无动作的时隙仍不给交付。11 月 19 日旧执行器的完整闭环还有请求时机、插值、预热时钟和前沿实现等差别，单项试验不能替代整个旧执行器。')
para(f'更严格的原状态复核直接读取第六章双方原始预热状态，逐世界核验初始观测时间、原始量测、位置、协方差及径向速度等的 SHA-256；轨迹、速度、量测噪声及阴影直接取自同一旧缓存。固定双方旧权重、专用头、原乘子、5 dB 后，同一批四个旧世界的 C−RC 从原程序 {sg(exact.loc["C","archive_G"]-exact.loc["RC","archive_G"])} 变为现程序 {sg(exact.loc["C","modern_G"]-exact.loc["RC","modern_G"])} Mbps。此数识别整个执行平台的合计效应，不能全归于动作到期。','small')
new('路线记忆、IMM 与门控能解释多少')
para('同一批四世界、紧档、0 dB、价值调度。以下各系统使用自己的归档误差头和乘子；改变中心与头后的闭环吞吐见表。')
rr=[['中心与门控','各自头吞吐 / 风险','全用 RC 头吞吐 / 风险','各自头有效窄束秒']]
for name,cn in [('I','IMM（I）'),('P','路线专家（P）'),('RC','路线专家与 IMM 的 G1 门控')]:
    v=row(name);b=row(name,'same_RC') if name!='RC' else v;rr.append([cn,f'{v.G:.2f} / {v.risk_pct:.2f}%',f'{b.G:.2f} / {b.risk_pct:.2f}%',fmt(v.narrow_s)])
table(rr,[175,119,132,89])
para(f'三臂全用归档 RC 的同一个头和同一组乘子，唯一主动切换的是中心；闭环观测随动作自然分化。此时 I→P 吞吐变化 {sg(row("P","same_RC").G-row("I","same_RC").G)} Mbps，P→G1 融合变化 {sg(row("RC").G-row("P","same_RC").G)} Mbps。该控制特意固定了 RC 的工作点，迁移至其他中心的风险也要同时读。','small')
para('进一步隔离门控。取 40 世界、640 架次锚点的最后共同预热观测，固定 I 和 P 给出的两个中心。用真值为每个锚点、每个期限单独选择 I 与 P 的凸混合权重。三维位置使用线段上的精确最小值，双角误差使用 0.001 步长的权重网格，并补入实际门控权重及位置最优权重。两种事后最优权重可不同，线上无法利用真值。')
rr=[['双角误差 °','IMM','路线 P','实际 RC','事后最优混合','微调 Chronos']]
for hz in (1,3,5,10):rr.append([f'{hz} 秒']+[fmt(H.loc[(m,hz),'angle_deg']) for m in ('I','P','RC','oracle','tuned')])
table(rr,[86,67,75,82,110,95])
para('5 秒的 RC 相对微调 Chronos 差约 0.64°；即便逐锚点事后挑权重，仍相差约 0.21°。10 秒事后混合与微调 Chronos 仍相差约 0.65°。短期有门控可改进空间，较长期还需改善至少一个专家可提供的预测中心。事后角误差不能换算成可实现的吞吐增益。')
para('旧版 10 个固定历史紧档世界中，直接给冻结 Chronos 加入路线协变量后，CR_raw 相比 C 下降 24.17 Mbps；在全生命周期稀疏预热下，同一增量为 +28.92 Mbps。11 月 21 日另有已训练的 Chronos 路线＋IMM 门控：紧档 5 dB、70 世界中，CR_raw 为 639.03，加入门控并重训头后的 CR-G 为 834.65，RC 为 832.96 Mbps。该批增量同时含门控与配套头，不能单记在门控名下。')
para('11 月 19 日的独立紧档验证（20 世界）中，固定 local 头规格与按期限校准，G0 换 G1 并重拟合配套头，吞吐增加 37.93 Mbps；固定 G1 与校准，local 头换 full 头增加 68.49 Mbps；固定 G0，仅改全局为按期限校准增加 95.78 Mbps。RC 的 G1 根据归一化误差和历史数据时间权重训练，当前两轮沿用归档门控。它影响 RC 水平；现有数据没有训练“直接最大化吞吐”的新门控。')
new('复核范围与数据入口')
para('本报告对排名的解释优先使用同世界、同输入、单项替换的对照。跨复现批次的绝对吞吐变化同时受执行器和环境影响。平均差值难以唯一分给若干模块，尤其是检查点与原生分位管发生交互时。')
para('对照与拟合边界','sub')
para('旧 C 的复现接口用冻结的本地官方 Chronos-2。旧 C 归档头试验使用第六章 C 的 full LightGBM 和候选期限乘子；按新执行器取共享因果特征并执行闭环。新外接头控制实验使用相同特征、损失、训练折和校准路线，每个中心分别拟合。种子 7 已发布的微调权重、原生管校准及外接树直接复用。')
para('输入与训练的同历史诊断以任务约第 300 秒为锚点，保留长历史的真实支持。门控事后对照以共同预热末尾为锚点，因此二者的 5 秒误差有不同绝对值。正式与补充仿真都使用合成无线链路；这轮归因未再次运行实飞回放，不能据这些效应给实飞排名分摊原因。')
para('复跑一致性','sub')
para('MCS 专项的 0 dB Chronos 重跑与四场正式归档的吞吐一致。RC 重跑相对正式归档在世界 0 高 7.05 Mbps，其余世界相差小于 0.01 Mbps；世界 0 和世界 20 的起始中心哈希与归档有差异。余量效应用本次同实现配对的 0 和 5 dB 结果，未把归档与新跑的数值混作一对。文件保存逐世界收益、模型版本、误差管和参数。')
para('旧世界 7 的 C 原状态对照在一次预测内复用重复路线弧长与分支计算，保持候选遍历次序、分支权重和决策规则。同一实现对旧世界 0 的独立复跑在吞吐、风险及放宽结算结果上与逐次展开完全一致。核验回执保存在“复算校验”目录。','small')
para('对应文件','sub')
for a in ['跨版本归因证据.xlsx　配对因素效应、各配置均值、96 场闭环、16 场 MCS、8 场原状态跨执行器、同历史逐锚点及旧版归档对照。','配对因素效应.csv　各对照的均值、世界内最小和最大变化，以及各世界具体变动。','closed-loop 原始记录位于本目录 cells/。MCS 余量配对目录保留 0/5 dB 的独立复跑。compare.py、matched_rc.py、anchors.py、gate_oracle.py 提供逐场仿真与固定历史入口。','历史证据为本地 RouteClock_第六章完整实验_20261119/数据表，GitHub 发布标签 ch6-fixed-history-20261119。本轮正式归档为 routeclock_runs/confirm_0db/results/summary。']:para(a,'small')
dst=O/'RC与Chronos跨版本归因报告.pdf'
SimpleDocTemplate(str(dst),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=36,bottomMargin=46).build(s,onFirstPage=footer,onLaterPages=footer)
