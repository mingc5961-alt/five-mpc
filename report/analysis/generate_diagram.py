from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
O=Path(__file__).resolve().parents[1];font=FontProperties(fname=O/'assets/NotoSansSC-Regular.ttf')
fig,ax=plt.subplots(figsize=(12,5.8));ax.set(xlim=(0,12),ylim=(0,5.8));ax.axis('off')
def box(x,y,w,h,text,color='#e7f1f2'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#397781',linewidth=1.2));ax.text(x+w/2,y+h/2,text,fontproperties=font,fontsize=10,ha='center',va='center',linespacing=1.7)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#52677b',linewidth=1.2))
box(.15,3.5,1.6,1.1,'过去带噪观测\n只用因果历史');box(2.1,3.5,2,1.1,'IMM 短期外推\n空间路线 + 进度');box(4.5,3.5,1.7,1.1,'时域门控融合\nRC 指向中心');box(6.6,3.5,2.1,1.1,'误差尺度头 + 校准\n整段双轴包络');box(9.1,3.5,2.6,1.1,'9 种束宽 × 7 个间隔\n可行候选与预期速率')
for a,b in [((1.75,4.05),(2.1,4.05)),((4.1,4.05),(4.5,4.05)),((6.2,4.05),(6.6,4.05)),((8.7,4.05),(9.1,4.05))]:arrow(a,b)
box(8.7,1.1,3,1.25,'FIVE：共享预算选动作\nMPC：风险 + 多步刷新/通信\n两条独立实验线','#e8ecf5');arrow((10.4,3.5),(10.4,2.35))
box(4.5,1.1,3.4,1.25,'实际传输：指向 / 增益 / MCS\n扣除感知与切换时间\n记录吞吐、可靠性和公平性','#f4eddf');arrow((8.7,1.7),(7.9,1.7))
box(.15,1.1,3.2,1.25,'按调度获得下一次真实观测\n更新历史与预测包\n形成闭环');arrow((4.5,1.7),(3.35,1.7));arrow((.95,2.35),(.95,3.5))
ax.text(6,5.35,'核心链条：预测信息必须经过不确定性与动作选择，才能变成通信收益',fontproperties=font,fontsize=14,ha='center',color='#005e68')
ax.text(6,.35,'轨迹真值只用于生成观测、结算和事后评分；不作为预测器的真实未来输入。',fontproperties=font,fontsize=10,ha='center',color='#52677b')
for ext in ['png','svg']:fig.savefig(O/f'figures/14_mechanism.{ext}',dpi=180,bbox_inches='tight')
# Presentation refinement: avoid overlapping labels for the identical global controls.
import pandas as pd,numpy as np
plt.close('all');plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
v=pd.read_csv(O/'tables/offline_validity.csv');v=v[(v.method=='RC')&(v.seed==7)]
fig,axs=plt.subplots(1,2,figsize=(11,4.2))
for hd,mk,color in [('lightgbm','o','#007e87'),('tabpfn','s','#7d62a3')]:
 for _,r in v[v.error_model==hd].iterrows():
  label=('LGB' if hd=='lightgbm' else 'TabPFN')+' '+r.variant
  if np.isfinite(r.mean_halfwidth_deg) and not(hd=='tabpfn' and r.variant=='global'):
   axs[0].scatter(r.mean_halfwidth_deg,r.covered*100,marker=mk,color=color)
   txt='global (both heads)' if r.variant=='global' else label
   axs[0].annotate(txt,(r.mean_halfwidth_deg,r.covered*100),xytext=(5,5),textcoords='offset points',fontsize=8)
  if np.isfinite(r.early_failure_given_issued):
   axs[1].scatter(r.approved_s,r.early_failure_given_issued*100,marker=mk,color=color)
   axs[1].annotate(label,(r.approved_s,r.early_failure_given_issued*100),xytext=(5,5),textcoords='offset points',fontsize=8)
axs[0].axhline(90,ls='--',color='gray');axs[0].set(xlabel='Mean tube halfwidth (degree)',ylabel='Whole-interval coverage (%)',xlim=(4.5,8.2),ylim=(89.8,93.4));axs[1].set(xlabel='Mean approved duration (s)',ylabel='Early failure among issued (%)',xlim=(1.4,2.85),ylim=(.35,3.9));plt.tight_layout()
for ext in ['png','svg']:fig.savefig(O/f'figures/08_validity.{ext}',dpi=180,bbox_inches='tight')
