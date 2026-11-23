from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np,pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
p=Path('/data/chen/refine-logs/RC_Chronos2_差距定位')
pdfmetrics.registerFont(TTFont('SC','/data/chen/teacher_report_20261118/assets/NotoSansSC-Regular.ttf'))
doc=SimpleDocTemplate(str(p/'RC与Chronos-2_差距定位.pdf'),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=40,bottomMargin=45)
story=[]
def para(s,size=11,colour='#193038',after=8):
 return Paragraph(escape(str(s)),ParagraphStyle('text',fontName='SC',fontSize=size,leading=size*1.52,textColor=colors.HexColor(colour),wordWrap='CJK',spaceAfter=after))
def text(s):story.append(para(s))
def heading(title,sub,first=False):
 if not first:story.append(PageBreak())
 story.extend([para(sub,9,'#537278',4),para(title,20,'#0c535c',12)])
def h(s):story.append(para(s,13,'#09646b',5))
def table(hd,body,w):
 rows=[[para(s,9,'#ffffff' if i==0 else '#193038',0) for s in row] for i,row in enumerate([hd]+body)]
 t=Table(rows,colWidths=[x*(doc.width-12) for x in w],hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#18575d')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf3f2')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#9baeb0'))]));story.extend([t,Spacer(1,8)])
heading('RC 的差距从何时开始？','RouteClock　/　0 dB 首版报告补充诊断　·　2026 年 11 月 23 日',True)
text('比较对象为路线辅助中心＋归档误差头的 RC，以及全量微调中心＋原生分位误差管的 Chronos‑2。先固定观测、比较同一预测期限，再检查同一窄波束的物理指向与模型准入，最后回到闭环实际交付。')
h('同一观测历史，当前估计与未来预测呈现不同结果')
text('在全部 40 个合成确认世界（每世界 16 机）和全部 59 个实飞架次上，取最后一次共同预热观测作为起点。先进中心固定种子 7。表中每格为 RC / Chronos‑2，位置用三维欧氏距离，双角误差用环绕后的方位、俯仰误差向量长度；按世界或架次等权。')
a=pd.concat([pd.read_csv(f) for f in sorted(p.glob('anchors_*.csv'))])
def dual(g,col,n=2,factor=1):
 v=g.groupby('model')[col].mean()*factor
 return f'{v["RC"]:.{n}f} / {v["Chronos-2"]:.{n}f}'
rows=[]
for domain in ('合成','实飞'):
 for horizon in (0,1,3,5,10):
  z=a[(a.domain==domain)&(a.h==horizon)]
  rows.append([domain,f'{horizon} 秒',dual(z,'position_m'),dual(z,'angle_deg'),dual(z,'cert32',1,100) if horizon else '—'])
table(['环境','预测年龄','位置误差（米）','双角误差（度）','32×32 准入（%）'],rows,[.13,.16,.23,.23,.25])
text('两域的当前三维位置误差都以 RC 更小，实飞当下的双角误差则以 Chronos‑2 更小。两种指标取决于误差方向，不能互相替代。到第 5 秒，RC 多出的三维误差为合成 4.23 米 [3.46, 5.04]、实飞 4.67 米 [3.09, 6.41]（配对 95% 区间）。在实飞三角航线，5 秒位置误差为 13.50 / 3.67 米；矩形为 3.84 / 1.81 米，优先排查三角航线中期外推。')
text('范围　每架次此项只有一个预热末锚点。合成首预测与原档逐字节哈希一致的世界为 RC 18/40、Chronos‑2 20/40；实飞两法均为 59/59。合成缓存观测位置与重读值在 10⁻⁹ 米内核对，再将相同缓存观测同时交给两法。均值代表本次共同输入复算。')
heading('预测中心与误差管，分别卡在哪里？','机制核对　/　固定阵列与预测期限')
text('固定 32×32 和 5 秒期限，用 20 ms 真值网格判断整段几何损失是否始终 ≤3 dB；准入则按原调度前沿的校准误差包络检查。实飞 59 个预热末锚点中，RC 的中心有 31 个实际上可保持 5 秒，误差管放行 0 个；Chronos‑2 有 52 个实际可保持，放行 44 个，其中 42 个实际通过。RC 既有中心失准，也留下了一批本可用窄束的机会。')
text('合成 640 个机级锚点中，RC / Chronos‑2 的实际可保持比例为 73.1% / 85.0%，准入比例为 30.0% / 56.9%。实飞 5 秒端点，RC 的平均校准方位/俯仰半宽为 6.81°/2.44°，Chronos‑2 为 1.25°/0.44°。半宽是误差管输出，判断准入还要检查整个预测前缀。')
h('在服务期实际付费观测后，把中心和误差管交叉替换')
text('T 档、种子 7，事先按编号抽取合成世界 0/10/20/30、两类实飞航线各取清单首/中/末架次。每机取服务开始 20、70、120 秒后的首次 RC 付费观测，给两法同一份 RC 历史。下表只比较 5 秒准入；“错放”是全部锚点中被放行且按该中心的真值几何损失曾超过 3 dB 的占比。')
s=pd.concat([pd.read_csv(f) for f in sorted(p.glob('service_*.csv'))]);s=s[s.h==5].copy()
s['false_accept']=s.cert32&~s.safe32_prefix
rows=[]
for domain in ('合成','实飞'):
 for ctr,width in [('RC','RC'),('Chronos-2','RC'),('RC','Chronos-2'),('Chronos-2','Chronos-2')]:
  z=s[(s.domain==domain)&(s.center==ctr)&(s.width==width)]
  v=z.groupby('unit')[['err_m','cert32','false_accept']].mean().mean()
  rows.append([domain,ctr,width,f'{v.err_m:.2f}',f'{v.cert32*100:.1f}%',f'{v.false_accept*100:.1f}%'])
table(['环境','中心','误差管','位置误差 m','获准 5 秒','错放占比'],rows,[.12,.19,.19,.18,.16,.16])
text('同一中心换管时，位置预测完全相同，窄束准入却大幅改变。只换中心、保留 RC 管，准入变化很小。与此同时，两域服务期样本的 RC 中心 5 秒误差仍更大。交叉管沿用原校准乘子，缺少对新中心的独立校准；其准入率用于辨认控制瓶颈，不能读成新方案的吞吐或风险保证。')
heading('窄波束最终贡献了多少通信收益？','业务核对　/　正式闭环与抽样逐束分账')
text('两法共用 0 dB MCS 余量、价值调度、令牌桶和 10 秒最长持有。合成 16 机共享、实飞单机分摊，T/M/L 三档 180 秒服务感知额度为合成 3/4.5/9 秒、实飞 0.1875/0.28125/0.5625 秒。以下正式量覆盖全部世界/架次，合成先聚合三个训练种子。')
table(['场景','预算','净吞吐差 Mbps','有效时间贡献','段内速率贡献'],[
 ['合成','T','+171.9','+2.9','+169.0'],['合成','M','+101.2','+15.3','+85.9'],['合成','L','+35.5','+15.4','+20.1'],
 ['实飞','T','+161.8','+103.5','+58.3'],['实飞','M','+11.3','+2.6','+8.7'],['实飞','L','+3.3','+1.9','+1.3']],[.14,.1,.26,.25,.25])
text('差值方向为 Chronos‑2 减 RC。对每一对闭环按 G=有效时间×段内速率/180 作对称分解，再平均，显示值按位数取整。合成 T 档有效 32×32 时间由 40.47 增至 88.53 秒；实飞 T 档由 4.12 增至 46.86 秒。实飞有效发送时间同时从 167.05 增至 178.10 秒，RC 的可用动作不足也影响其他阵列的发包。')
h('重放按实际采用的阵列记账')
b=pd.concat([pd.read_csv(f) for f in sorted(p.glob('beam_*.csv'))])
rows=[]
for domain in ['合成','实飞']:
 v=b[b.domain==domain].groupby('model')[['G32','Gother','replay_G']].mean()
 for m in ['RC','Chronos-2']:rows.append([domain,m,*[f'{y:.1f}' for y in v.loc[m]]])
 rows.append([domain,'两法差值',*[f'{y:+.1f}' for y in (v.loc['Chronos-2']-v.loc['RC'])]])
table(['T 档抽样','方法','32×32 交付','其他阵列交付','合计 Mbps'],rows,[.16,.19,.22,.23,.2])
dg=(b.replay_G-b.original_G).abs().max();dR=100*(b.replay_R-b.original_R).abs().max()
text(f'两列均为对应成功交付比特除以完整 180 秒，二者相加等于总吞吐。窄束交付增加的同时，其他阵列的交付减少，所以单列增量不能当成净增量。此表仅为 4 世界/6 架次抽样重放；与原场记录逐一比对，最大差异为 {dg:.3f} Mbps、总风险 {dR:.3f} 个百分点。正式差值仍取全量原档。')
text('改进优先级　先在相同历史上改善 RC 的 3–5 秒双角外推，再按中心、期限和路线支持状态独立校准误差管。在同一风险水平比较“真实可保持且获准”的窄束比例和净交付；直接把整条管缩窄会同时放行一部分实际不可保持的动作。')
def footer(c,d):
 c.setFont('SC',8);c.setFillColor(colors.HexColor('#577178'));c.drawString(44,25,'RouteClock　模型差距补充诊断');c.drawRightString(A4[0]-44,25,str(d.page))
doc.build(story,onFirstPage=footer,onLaterPages=footer)
