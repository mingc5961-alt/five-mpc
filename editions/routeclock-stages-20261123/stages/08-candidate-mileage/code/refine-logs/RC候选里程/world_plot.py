import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from protocol_local import OUT
def draw(data,method):
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.6),layout='constrained')
    for ax,h in zip(axes,(5,10)):
        for key,name,color,shape in [('prior','Worlds 40–49','#999da0','x'),('fresh','Primary 50–59','#007663','o'),('replicate','Replication 60–69','#ac5941','^')]:
            d=data[data['批次']==key];ax.scatter(d[f'Chronos-2_{h}'],d[f'{method}_{h}'],color=color,label=name,marker=shape,s=32,alpha=.9)
        m=1.05*max(data[f'Chronos-2_{h}'].max(),data[f'{method}_{h}'].max())
        x=np.linspace(0,m,100);ax.plot(x,x,color='#252c35',lw=1,label='Equal error')
        ax.fill_between(x,x,1.10*x,color='#afc7b5',alpha=.4,label='Up to 10% higher')
        ax.set(xlim=(0,m),ylim=(0,m),aspect='equal',title=f'{h}-second horizon',xlabel='Chronos-2 position error (m)',ylabel='New RC position error (m)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8,loc='upper left',frameon=False)
    dst=OUT/'逐世界位置误差.svg';fig.savefig(dst);plt.close(fig);s=dst.read_text();return s[s.index('<svg'):].replace('<svg ','<svg style="width:100%;height:auto" ',1)
