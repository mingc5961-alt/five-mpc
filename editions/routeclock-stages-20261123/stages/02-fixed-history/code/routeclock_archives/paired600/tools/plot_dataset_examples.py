"""Plot saved test forecasts; reconstruct missing observation history deterministically."""
import sys,json,pickle,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from routeclock.world import World,GROUPS,rng_for
from routeclock.dataset import observation_schedule

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/dataset_visual_examples'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10})
cfg=json.loads((ROOT/'data/final_real/configuration.json').read_text(encoding='utf-8'))
with (ROOT/'runs/final_tabpfn_models/test_predictions.pkl').open('rb') as f:records=pickle.load(f)
with (ROOT/'data/final_real/test.pkl').open('rb') as f:original=pickle.load(f)
labels=['稳定路径＋稳定节奏','稳定路径＋整体变速','稳定路径＋分段变速','稳定路径＋分段变速与停留','局部偏离后归位','无可复用路径机动']
methods={'I':'运动模型 IMM','P':'空间路径＋进度','H':'时间周期模型','C':'Chronos（不输入路线）','CR_raw':'路线辅助 Chronos（未融合）','RC':'空间路径与 IMM 融合','CR':'路线辅助 Chronos 与 IMM 融合'}
colors=dict(zip(methods,['#0072B2','#E69F00','#AA4499','#009E73','#56B4E9','#D55E00','#CC3377']))
styles={'I':'--','P':'--','H':':','C':'-.','CR_raw':':','RC':'-','CR':'-'}
chosen=[];metrics=[];manifest=[]

def base(ax,hist,truehist,truth,origin):
    ax.plot(*(truehist[:,:2]-origin[:2]).T,color='#aaaaaa',lw=1,label='历史真实轨迹（仅展示）')
    ax.plot(*(hist[:,:2]-origin[:2]).T,'o:',color='#555555',ms=3,lw=.8,label='历史带噪观测')
    ax.plot(*(truth[:,:2]-origin[:2]).T,color='black',lw=2.4,label='真实未来（10秒）')
    ax.scatter([0],[0],marker='*',s=130,c='black',zorder=8,label='预测起点真实位置')
    ax.set_xlabel('相对起点 X / 米');ax.set_ylabel('相对起点 Y / 米')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)

for group,title in zip(GROUPS,labels):
    candidates=[a for a in records if a['group']==group]
    task=min({a['task'] for a in candidates},key=lambda s:int(s.rsplit('/',1)[1]))
    a=max((a for a in candidates if a['task']==task),key=lambda a:a['package'].origin)
    pkg=a['package'];namespace,mission=task.rsplit('/',1);mission=int(mission)
    source=next(x for x in original if x['task']==task and x['package'].origin==pkg.origin)
    for m in source['package'].centers:np.testing.assert_array_equal(source['package'].centers[m],pkg.centers[m])
    world=World(cfg,namespace,mission,n=1,groups=[group])
    anchors=[cfg['simulation']['warmup_s']*v for v in [.65,.82,1.]]
    r=rng_for(namespace,mission,777)
    times=observation_schedule(anchors,anchors,[.5,1,2,5,10,None][mission%6],r)
    times=np.array([t for t in times if t<=pkg.origin])
    hist=np.array([world.observe(0,t).position for t in times])
    truehist=np.array([world.truth(0,t)[0] for t in np.arange(0,pkg.origin+.001,.02)])
    recreated=np.array([world.truth(0,pkg.origin+h)[0] for h in a['truth_h']])
    np.testing.assert_allclose(recreated,a['truth'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(hist[-1],pkg.centers['C'][0],rtol=0,atol=1e-10)
    origin=a['truth'][0];truth=a['truth'];chosen.append((a,title,hist,truehist,origin))
    fig,axs=plt.subplots(2,2,figsize=(16,12),constrained_layout=True)
    for ax,subset,subtitle in [(axs[0,0],['I','P','H','C','CR_raw'],'五种基础位置预测'),(axs[0,1],['I','P','CR_raw','RC','CR'],'两种学习融合与输入专家')]:
        base(ax,hist,truehist,truth,origin)
        for m in subset:ax.plot(*(pkg.centers[m][:,:2]-origin[:2]).T,styles[m],color=colors[m],lw=1.7,label=methods[m])
        ax.set_title(subtitle);ax.legend(fontsize=8,loc='best')
    ax=axs[1,0]
    ax.plot(times-pkg.origin,hist[:,2]-origin[2],'o:',c='#555555',ms=3,label='历史带噪观测')
    ax.plot(a['truth_h'],truth[:,2]-origin[2],c='black',lw=2.4,label='真实未来')
    for m in methods:ax.plot(pkg.hs,pkg.centers[m][:,2]-origin[2],styles[m],c=colors[m],label=methods[m])
    ax.axvline(0,c='gray',lw=.8);ax.set(xlabel='相对预测起点时间 / 秒',ylabel='相对起点高度 / 米',title='高度变化：负时间为历史，正时间为未来');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    ax=axs[1,1]
    predrows=[]
    for m in methods:
        pred=np.array([pkg.at(m,float(h)) for h in a['truth_h']]);error=np.linalg.norm(pred-truth,axis=1)
        ax.plot(a['truth_h'],error,styles[m],color=colors[m],label=methods[m])
        metrics.append({'behavior':title,'task':task,'method':methods[m],'code':m,'ADE_m':float(error.mean()),'FDE_10s_m':float(error[-1])})
        for h,p,y in zip(a['truth_h'],pred,truth):predrows.append([m,float(h),*p,*y])
    ax.set(xlabel='预测时长 / 秒',ylabel='三维位置误差 / 米',title='三维位置误差（所有方法，同一真实未来）');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.suptitle(f'{title}  |  测试任务 {mission}  |  历史 {pkg.origin:g} 秒，预测未来 10 秒\n按类别选最小任务编号的最后预测时刻；类别为生成机制，不代表窗口内必然发生停留或偏离',fontsize=14)
    stem=f'{mission+1:02d}_{group}'
    fig.savefig(OUT/f'{stem}.png',dpi=160);plt.close(fig)
    with (OUT/f'{stem}_observations.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['time_s','x_m','y_m','z_m']);w.writerows(np.c_[times,hist])
    with (OUT/f'{stem}_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['method','horizon_s','pred_x','pred_y','pred_z','truth_x','truth_y','truth_z']);w.writerows(predrows)
    manifest.append({'task':task,'group':group,'origin_s':pkg.origin,'observations':len(times),'history_source':'deterministic replay from saved configuration and generator; future truth and latest observation verified','file':stem+'.png'})

fig,axs=plt.subplots(2,3,figsize=(18,11),constrained_layout=True)
for ax,(a,title,hist,truehist,origin) in zip(axs.flat,chosen):
    base(ax,hist,truehist,a['truth'],origin)
    for m in ['I','RC','CR']:
        ax.plot(*(a['package'].centers[m][:,:2]-origin[:2]).T,styles[m],color=colors[m],lw=1.7,label=methods[m])
    ax.set_title(title);ax.legend(fontsize=7,loc='best')
fig.suptitle('当前测试集六类行为概览：12秒历史 → 10秒未来\n主方法水平投影；全部七种预测、高度与误差见各类别详图',fontsize=16)
fig.savefig(OUT/'00_six_behaviors_overview.png',dpi=160);plt.close(fig)
with (OUT/'sample_errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
notes='''# 当前数据集轨迹样例

先看 `00_six_behaviors_overview.png`，然后按编号打开六张分类详图。各详图含基础预测、融合预测、高度、三维位置误差四个面板。黑线是真实未来，灰色圆点是历史带噪观测，灰色细线是历史真值（仅用于展示，不输入预测器）；彩色线是保存的模型预测。所有位置相对该预测起点的真实位置平移，单位米。

样本来自 `data/final_real/test.pkl`；融合预测来自 `runs/final_tabpfn_models/test_predictions.pkl`。每类按最小任务编号，选择最后一个预测时刻，不按预测效果筛选。原始五种预测与数据集逐元素核对一致，没有重新训练或重新推理。历史观测未单独保存在原始pkl中，因此按同一生成配置、任务随机种子与观测时序重建；重建的未来真值与已保存真值、最后观测与Chronos输入起点均核对通过。

六类是仿真生成机制：稳定路径稳定节奏、整体变速、分段变速、分段变速与停留、局部偏离后归位、无可复用路径机动。12秒历史＋10秒未来的窗口可能未包含停留、偏离或完整一圈，不能从类别名称断言图中已经发生这些事件。各任务采用不同的既定观测间隔；同一任务内所有方法使用同一批观测。

这些图展示单次发布后的10秒开环预测，不在未来中途追加观测，因此与会不断更新预测的通信闭环含义不同。部分模型外推很远时保留完整坐标范围，不裁去错误；三维误差图可用于查看实际偏差。

`sample_errors.csv`保存42组样例误差，ADE为整个未来窗口平均三维位置误差，FDE为10秒末端误差。各类别另有观测CSV、预测与真值CSV。`manifest.json`记录样本编号与来源。
'''
(OUT/'README.md').write_text(notes,encoding='utf-8')
print(OUT)
print(json.dumps(metrics,ensure_ascii=False))
