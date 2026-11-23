from io import StringIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import ROOT
def figure(centers,pairs):
    fig,(a,b)=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
    for key,label,col,style in [('VECTOR','VECTOR','#9c713b',':'),('RAFT','RAFT','#756b96','-.'),('RC','Original RC','#495867','--'),('joint','RC with learned progress and paths','#008164','-'),('Chronos-2','Chronos-2','#a53545','-')]:
        sub=centers[centers.arm==key].sort_values('horizon')
        a.plot(sub.horizon,sub.position_m,style,color=col,label=label,lw=2,marker='.',ms=6)
    a.set(xlabel='Forecast horizon (s)',ylabel='Mean endpoint error (m)',title='Same observation history in 10 new worlds')
    a.legend(fontsize=8,frameon=False);a.set_ylim(bottom=0)
    for j,(name,label,col) in enumerate([('joint-transfer','New center, original tube','#8da6a0'),('joint-refit','New center, refitted tube','#008164')]):
        r=pairs[(pairs['模型']==name)&(pairs['对照']=='RC')&(pairs['指标']=='G')].set_index('预算').loc[['T','M','L']]
        x=np.arange(3)+(j-.5)*.32
        b.bar(x,r['均差'],width=.30,color=col,label=label)
        b.errorbar(x,r['均差'],np.stack([r['均差']-r['下限'],r['上限']-r['均差']]),color='#24323c',capsize=3,fmt='none')
    b.axhline(0,color='#24323c',lw=.8);b.set_xticks(range(3),['T (3 s)','M (4.5 s)','L (9 s)'])
    b.set(xlabel='Sensing budget over 180 s',ylabel='Paired goodput change from original RC (Mbps)',title='Closed-loop impact of center and tube')
    b.legend(fontsize=8,frameon=False)
    for ax in (a,b):
        ax.grid(axis='y',color='#d6dedc',alpha=.6);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    dst=ROOT/'中心误差与吞吐配对差.svg';fig.savefig(dst);plt.close(fig)
    s=dst.read_text();return s[s.index('<svg'):].replace('<svg ','<svg style="width:100%;height:auto" ',1)
