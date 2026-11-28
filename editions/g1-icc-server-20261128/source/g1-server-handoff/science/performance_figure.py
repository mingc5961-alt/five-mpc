"""Static paired-effect figure for the supplementary Linux holdout."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from server_paths import RUN

folder=RUN/'analysis'
panel=pd.read_csv(folder/'panel_summary.csv')
closed=pd.read_csv(folder/'closed_contrasts.csv')
fig,axes=plt.subplots(1,2,figsize=(10.2,3.55),layout='constrained')
palette=['#226654','#4178a0','#735a99','#a36934']


def show(ax,rows,labels,xlabel):
    for i,(row,label) in enumerate(zip(rows,labels)):
        y=len(rows)-1-i
        ax.errorbar(row['mean'],y,xerr=[[row['mean']-row['lo']],[row['hi']-row['mean']]],
                    fmt='o',capsize=3,ms=6,color=palette[i])
    ax.set_yticks(np.arange(len(rows))[::-1],labels,fontsize=9)
    ax.axvline(0,color='#596168',lw=.85,ls='--')
    ax.grid(axis='x',alpha=.2)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-.6,len(rows)-.4)
    ax.spines[['right','top','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)


pr=panel[(panel.label=='same_direction_revisit')&(panel.exposure=='imposed')].set_index('metric')
show(axes[0],[pr.loc[k] for k in ['d_M_minus_A','d_Fstar_minus_A','d_IMM_minus_RAFT','d_no_age_minus_with_age']],
      ['Motion − adaptive','Fixed fusion − adaptive','IMM − RAFT','Snapshot (no age − age)'],
      'Paired difference in LOS error (degrees)')
cr=closed[(closed['mode']=='common075')&(closed.metric=='G_Mbps')].set_index(['left','right'])
show(axes[1],[cr.loc[k] for k in [('P03','P_M'),('P03','P_Fstar'),('Q01','Q02'),('Q01','Q05')]],
      ['Oldest: adaptive − motion','Oldest: adaptive − fixed','Due: adaptive − motion','Due: adaptive − fixed'],
      'Paired difference in throughput (Mbps)')
axes[0].set_title('Same-direction revisits',loc='left',weight='bold',size=11)
axes[1].set_title('Equal-cost closed loop at 0.75',loc='left',weight='bold',size=11)
fig.savefig(folder/'paired_effects.png',dpi=180)
