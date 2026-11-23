import warnings
from pathlib import Path
from openpyxl import load_workbook
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from protocol import OUT,MODELS,BUDGETS

font='/data/chen/teacher_report_20261118/assets/NotoSansSC-Regular.ttf'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'pdf.fonttype':42,
                     'axes.unicode_minus':False,'font.size':9})
navy='#15394B';teal='#006C75';muted='#47616F';pale='#F0F5F5';rule='#D0DBDE'
cn={'T':'紧','M':'中','L':'松'}
kind=['公共输入尺度管','专用输入尺度管']
refs={'RC':'RC 归档尺度管','Chronos-2':'原生分位数管','VECTOR':'固定管','RAFT':'固定管'}
def header(title,sub):
    fig=plt.figure(figsize=(10.5,8.3),facecolor='white')
    fig.text(.065,.935,title,fontsize=21,color=navy,va='top',weight='bold')
    fig.text(.067,.86,sub,fontsize=10,color=muted,va='top',linespacing=1.65)
    return fig
def footer(fig,n):
    fig.text(.066,.033,'RouteClock　|　专用尺度管实验　·　种子 7',fontsize=8,color=muted)
    fig.text(.93,.033,f'{n} / 3',ha='right',fontsize=8,color=muted)
def table(fig,box,labels,rows,widths):
    ax=fig.add_axes(box);ax.axis('off')
    tb=ax.table(cellText=rows,colLabels=labels,colWidths=widths,cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    tb.auto_set_font_size(False);tb.set_fontsize(9)
    for (r,c),cell in tb.get_celld().items():
        cell.set_linewidth(.55);cell.set_edgecolor('#FFFFFF')
        cell.set_facecolor(navy if r==0 else (pale if r%2 else '#FAFCFC'))
        cell.set_text_props(color='white' if r==0 else navy)
    return tb
def rows(s):
    return list(s.values)[1:]
def render():
    book=load_workbook(OUT/'八个专用与公共尺度头实验结果.xlsx',read_only=True,data_only=True)
    d={(r[0],r[1],r[2]):r[5:] for r in rows(book['闭环均值'])}
    eff={(r[0],r[1],r[2],r[3],r[4]):r[5:] for r in rows(book['配对效应'])}
    offline={(r[0],r[1],r[4]):r[5:] for r in rows(book['独立验证'])}
    eq={(r[0],r[1],r[2]):r[7:] for r in rows(book['同覆盖率管宽'])}
    book.close()
    with PdfPages(OUT/'专用尺度管实验分析.pdf') as pdf:
        f=header('同一预测中心，换误差管后的闭环结果',
                 '每格为净吞吐 Mbps ／ 总风险 % ／ 实际感知秒数，取同一批十个开发世界的均值。\n16 个用户共享预算，预热和服务各 180 秒，MCS 余量 0 dB，价值调度，动作最长期限 10 秒。')
        tab=[]
        for m in MODELS:
            for k in (refs[m],*kind):
                series=[d[m,k,b] for b in BUDGETS]
                tab.append([m,k,*[f'{v[0]:.1f} ／ {v[1]:.2f} ／ {v[2]:.2f}' for v in series]])
        table(f,[.065,.23,.87,.525],['预测中心','误差管','紧档　上限 3.0 s','中档　上限 4.5 s','松档　上限 9.0 s'],tab,[.14,.18,.225,.225,.23])
        delta=[d['RC',kind[1],b][0]-d['RC',refs['RC'],b][0] for b in BUDGETS]
        cdelta=[d['Chronos-2',kind[1],b][0]-d['Chronos-2',refs['Chronos-2'],b][0] for b in BUDGETS]
        sig=lambda xs:' / '.join(f'{x:+.1f}' for x in xs)
        f.text(.067,.169,f'紧／中／松档，RC 专用新管相对归档 RC 的吞吐变化为 {sig(delta)} Mbps。',color=navy,fontsize=11)
        f.text(.067,.115,f'Chronos-2 专用新管相对自身原生分位数管的吞吐变化为 {sig(cdelta)} Mbps。',color=navy,fontsize=11)
        footer(f,1);pdf.savefig(f);plt.close(f)
        f=header('专用输入带来了多少增益',
                 '左图为专用输入减公共输入的世界内净吞吐差，横线为按世界成组重采样 5000 次的描述性 95% 区间。\n右表同向相减；风险按百分点计算，窄波束时间只累计 32×32 且几何损失 ≤3 dB 的有效数据时长。')
        ax=f.add_axes([.19,.18,.31,.55])
        labels=[];values=[];lo=[];hi=[];tab=[]
        for m in MODELS:
            for b in BUDGETS:
                a=eff[m,kind[1],m,kind[0],b]
                labels.append(f'{m}　{cn[b]}');values.append(a[0]);lo.append(a[1]);hi.append(a[2])
                tab.append([f'{m} {cn[b]}',f'{a[3]}/10',f'{a[4]:+.2f}',f'{a[9]:+.2f}',f'{a[10]:+.1f}'])
        ax.axvline(0,color=muted,linewidth=1)
        for i,(v,l,u) in enumerate(zip(values,lo,hi)):
            ax.plot([l,u],[i,i],color=teal,lw=1.4);ax.scatter([v],[i],color=teal,s=22,zorder=3)
        ax.set_yticks(range(12),labels);ax.set_ylim(11.5,-.7);ax.set_xlabel('净吞吐差（Mbps）');ax.grid(axis='x',color=rule,lw=.5);ax.spines[['top','right','left']].set_visible(False)
        table(f,[.53,.18,.41,.59],['模型/档','胜出','风险差','感知差 s','窄波束差 s'],tab,[.28,.15,.18,.18,.21])
        f.text(.067,.081,'风险由无有效动作和有效动作下的几何超限共同构成。两项分量及逐世界明细见配套工作簿。',color=muted)
        footer(f,2);pdf.savefig(f);plt.close(f)
        f=header('在相同观测历史上比较误差包络',
                 '400 条验证轨迹、1200 个锚点。覆盖率要求从发布至该期限的整个时间前缀内，方位和俯仰两轴同时落在管内。\n下表每格为实际覆盖率 % ／ 方位半宽 ° ／ 俯仰半宽 °；最后一列只用于同覆盖率下的宽度对比。')
        tab=[]
        for m in MODELS:
            for k in (refs[m],*kind):
                x,y=offline[m,k,5],offline[m,k,10]
                norm=eq.get((m,k,10))
                tab.append([m,k,*[f'{a[0]:.1f} ／ {a[1]:.2f} ／ {a[2]:.2f}' for a in (x,y)],
                            '—' if norm is None else f'{norm[0]:.2f} ／ {norm[1]:.2f}'])
        table(f,[.065,.255,.87,.49],['模型','误差管','5 秒实际','10 秒实际','10 秒同覆盖双角半宽'],tab,[.14,.19,.225,.225,.22])
        f.text(.067,.198,'训练与校准',color=navy,fontsize=12,weight='bold')
        f.text(.067,.156,'每个中心单独训练两棵树，训练集 240 条轨迹，调参按轨迹划分 192/48。独立 400 条校准轨迹逐期限校准。\n同覆盖诊断用验证集经验 90% 分位重新缩放，仅用于读表。闭环始终使用冻结的独立校准系数。\n单一种子、十个开发世界的闭环结果适用于本轮模拟条件。',color=muted,fontsize=9,va='top',linespacing=1.7)
        footer(f,3);pdf.savefig(f);plt.close(f)
if __name__=='__main__':render()
