"""Matched-world history-length pilot. Does not modify trained models or simulator."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,pickle,csv,copy,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from routeclock.world import World,GROUPS
from routeclock.predictors import Tracker,apply_gate
from routeclock.geometry import angles,angle_error
from routeclock.config import dump_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/history_scale_pilot'
SCALES=[12,30,60,120,240]
ORIGINS=[240.,300.]
LABELS=['稳定路径＋稳定节奏','稳定路径＋整体变速','稳定路径＋分段变速','分段变速与停留','局部偏离后归位','无可复用路径机动']
NAMES={'I':'运动模型 IMM','P':'空间路径＋进度','H':'时间周期模型','RC':'空间路径与 IMM 学习融合','HALF':'空间路径与 IMM 各占一半'}

def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def nearest_map_error(points,fragments):
    a=np.concatenate([f.points[:-1] for f in fragments]);b=np.concatenate([f.points[1:] for f in fragments]);d=b-a
    u=np.clip(np.einsum('nki,ki->nk',points[:,None,:]-a,d)/np.maximum(np.sum(d*d,axis=1),1e-12),0,1)
    dist=np.linalg.norm(points[:,None,:]-(a+u[:,:,None]*d),axis=2)
    return np.min(dist,axis=1)

def run():
    OUT.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'data/final_real/configuration.json').read_text(encoding='utf-8'))
    with (ROOT/'runs/final_tabpfn_models/bundle.pkl').open('rb') as f:gate=pickle.load(f)['gates']['RC']
    allrows=[];allrecords=[];manifest=[]
    for mission,(group,label) in enumerate(zip(GROUPS,LABELS)):
        namespace='offline/test';old=World(cfg,namespace,mission,n=1,groups=[group])
        extended=copy.deepcopy(cfg);extended['simulation']['warmup_s']=300
        world=World(extended,namespace,mission,n=1,groups=[group])
        # Changing simulation length moves the post-trajectory RNG noise draw.
        # Preserve original samples explicitly; reuse one long realization for every window.
        np.testing.assert_allclose(world.positions[0,:len(old.times)],old.positions[0],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(world.references[0],old.references[0])
        world.noise[0][:len(old.times)]=old.noise[0]
        obs_times=np.arange(0,301,.5);observations={float(t):world.observe(0,float(t)) for t in obs_times}
        np.savez_compressed(OUT/f'world_{mission}.npz',times=world.times,position=world.positions[0],velocity=world.velocities[0],reference=world.references[0],observation_times=obs_times,observations=np.array([o.position for o in observations.values()]))
        manifest.append({'task':f'{namespace}/{mission}','group':group,'original_physical_prefix_verified':True,'original_noise_prefix_preserved':True})
        for origin in ORIGINS:
            truth_h=np.arange(0,10.0001,.02);truth=np.array([world.truth(0,origin+h)[0] for h in truth_h])
            for length in SCALES:
                start=time.perf_counter();tracker=Tracker(extended)
                times=np.arange(origin-length,origin+.001,.5)
                for t in times:
                    tracker.observe(observations[float(t)]);pkg=tracker.issue()
                feat=tracker.features(pkg,'P');apply_gate(pkg,gate,feat,'RC','P')
                pkg.centers['HALF']=.5*(pkg.centers['P']+pkg.centers['I'])
                mem=tracker.memory;maperr=nearest_map_error(truth,mem.fragments)
                ref=world.references[0];ref_len=np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()
                progress=world.interpolate(world.progress[0],origin)-world.interpolate(world.progress[0],origin-length)
                diagnostics={'history_s':length,'origin_s':origin,'task':f'{namespace}/{mission}','group':group,'behavior':label,'n_observations':len(times),'fragments':len(mem.fragments),'closed_fragments':sum(f.closed for f in mem.fragments),'active_closed':mem.fragments[mem.active].closed,'revisits':mem.revisits,'map_length_m':sum(f.arc()[-1] for f in mem.fragments),'future_to_map_mean_m':float(maperr.mean()),'future_to_map_p95_m':float(np.quantile(maperr,.95)),'support_10s':float(pkg.diagnostics['support'][-1]),'progress_std_10s_m':float(pkg.diagnostics['progress_std'][-1]),'gate_route_weight_1s':float(gate.weights(feat[2:3,:16])[0]),'gate_route_weight_5s':float(gate.weights(feat[10:11,:16])[0]),'gate_route_weight_10s':float(gate.weights(feat[20:21,:16])[0]),'reference_length_m':float(ref_len),'generator_progress_traversals':float(progress/ref_len) if group!=GROUPS[-1] else None,'matching_residual':float(mem.innovation)}
                predictions={}
                for m in NAMES:
                    pred=np.array([pkg.at(m,float(h)) for h in truth_h]);predictions[m]=pred
                    errors=np.linalg.norm(pred-truth,axis=1);ae=np.linalg.norm(np.rad2deg(angle_error(angles(pred),angles(truth))),axis=1)
                    row={**diagnostics,'method':m,'method_name':NAMES[m]}
                    for horizon in [2,5,10]:
                        ix=truth_h<=horizon+1e-8
                        row[f'ADE_{horizon}s_m']=float(errors[ix].mean());row[f'FDE_{horizon}s_m']=float(errors[ix][-1]);row[f'angle_RMSE_{horizon}s_deg']=float(np.sqrt(np.mean(ae[ix]**2)))
                    allrows.append(row)
                record={'diagnostics':diagnostics,'history_times':times,'observations':np.array([observations[float(t)].position for t in times]),'history_truth_times':world.times[(world.times>=origin-length)&(world.times<=origin)],'history_truth':world.positions[0][(world.times>=origin-length)&(world.times<=origin)],'truth_h':truth_h,'truth':truth,'predictions':predictions,'memory_fragments':[{'points':f.points.copy(),'closed':f.closed} for f in mem.fragments],'gate_weights':gate.weights(feat[:,:16]),'hs':pkg.hs,'support':pkg.diagnostics['support'].copy()}
                allrecords.append(record)
                dump_json(OUT/'progress.json',{'finished':len(allrecords),'total':60,'latest':diagnostics})
                print(f'{label} origin={origin:g} history={length} closed={diagnostics["closed_fragments"]} support={diagnostics["support_10s"]:.2f} map_error={maperr.mean():.2f} P_ADE={allrows[-4]["ADE_10s_m"]:.2f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        # Persist per-world checkpoints; all histories for a world share its realized observations.
        csvwrite(OUT/'metrics.csv',allrows)
        with (OUT/'records.pkl').open('wb') as f:pickle.dump(allrecords,f)
    dump_json(OUT/'manifest.json',{'worlds':manifest,'history_scales_s':SCALES,'forecast_origins_s':ORIGINS,'observation_interval_s':.5,'forecast_horizon_s':10,'gate':'frozen existing final_tabpfn_models RC gate; no retraining','model_parameters_modified':False,'selection':'same first task of each group as prior figures; all windows at a given origin have identical future and nested observations'})
    dump_json(OUT/'configuration.json',extended)
    return allrecords,allrows

def plot(records,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':10})
    colors={'I':'#0072B2','P':'#E69F00','H':'#AA4499','RC':'#D55E00','HALF':'#009E73'}
    styles={'I':'--','P':'--','H':':','RC':'-','HALF':'-.'}
    for mission,(group,label) in enumerate(zip(GROUPS,LABELS)):
        for origin in ORIGINS:
            cases=[a for a in records if a['diagnostics']['group']==group and a['diagnostics']['origin_s']==origin]
            fig,axes=plt.subplots(5,3,figsize=(20,25),constrained_layout=True)
            for axrow,a in zip(axes,cases):
                d=a['diagnostics'];offset=a['truth'][0];true=a['truth']-offset;hist=a['history_truth']-offset;obs=a['observations']-offset
                ax=axrow[0];ax.plot(*hist[:,:2].T,c='#888888',lw=1,label='全部历史真值（仅展示）');ax.scatter(*obs[:,:2].T,c='#555555',s=5,label='全部历史观测')
                for k,f in enumerate(a['memory_fragments']):ax.plot(*(f['points'][:,:2]-offset[:2]).T,c='#56B4E9',lw=1.3,alpha=.8,label='已建立的路径记忆' if k==0 else None)
                ax.plot(*true[:,:2].T,c='black',lw=2.4,label='未来10秒真值');ax.scatter([0],[0],c='black',marker='*',s=100)
                ax.set_title(f'历史{d["history_s"]}秒／{d["n_observations"]}次观测；闭合片段{d["closed_fragments"]}\n未来真值到记忆路径平均距离：{d["future_to_map_mean_m"]:.2f}米')
                ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8,loc='best')
                ax=axrow[1];ax.plot(*true[:,:2].T,c='black',lw=2.4,label='真实未来');ax.scatter([0],[0],marker='*',c='black',s=100)
                for m in NAMES:ax.plot(*(a['predictions'][m][:,:2]-offset[:2]).T,styles[m],c=colors[m],lw=1.6,label=NAMES[m])
                ax.set_title(f'同一段未来：只改变可用历史\n10秒路线支持度{d["support_10s"]:.2f}／学习门控路线权重{d["gate_route_weight_10s"]:.2f}')
                ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8,loc='best')
                ax=axrow[2]
                for m in NAMES:ax.plot(a['truth_h'],np.linalg.norm(a['predictions'][m]-a['truth'],axis=1),styles[m],c=colors[m],lw=1.6,label=NAMES[m])
                ax.set(xlabel='向未来预测 / 秒',ylabel='三维位置误差 / 米',title='误差随预测时长变化');ax.legend(fontsize=8,loc='best')
                for ax in axrow[:2]:ax.set_xlabel('相对预测起点 X / 米');ax.set_ylabel('相对预测起点 Y / 米')
                for ax in axrow:ax.grid(alpha=.2)
            # Shared axes by column: no misleading resizing between history lengths.
            for col in range(3):
                xlo=min(ax.get_xlim()[0] for ax in axes[:,col]);xhi=max(ax.get_xlim()[1] for ax in axes[:,col]);ylo=min(ax.get_ylim()[0] for ax in axes[:,col]);yhi=max(ax.get_ylim()[1] for ax in axes[:,col])
                for ax in axes[:,col]:ax.set_xlim(xlo,xhi);ax.set_ylim(ylo,yhi)
            fig.suptitle(f'{label}｜任务{mission}｜固定在第{origin:g}秒预测未来10秒\n从上到下：12、30、60、120、240秒历史；观测间隔统一0.5秒，模型参数不变',fontsize=17)
            fig.savefig(OUT/f'{mission+1:02d}_{group}_t{int(origin)}.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
    for ax,group,label in zip(axes.flat,GROUPS,LABELS):
        for m in NAMES:
            y=[np.mean([r['ADE_10s_m'] for r in rows if r['group']==group and r['method']==m and r['history_s']==length]) for length in SCALES]
            ax.plot(SCALES,y,styles[m],marker='o',c=colors[m],label=NAMES[m])
        ax.set(title=label,xlabel='可用历史长度 / 秒',ylabel='未来10秒平均三维误差 / 米');ax.set_xticks(SCALES);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('历史变长后，预测是否改善？同一轨迹在240秒与300秒的两个预测窗口取平均\n固定同一未来、嵌套历史观测；每类一条轨迹，小规模机制诊断',fontsize=15)
    fig.savefig(OUT/'00_history_length_vs_error.png',dpi=160);plt.close(fig)

def report(records,rows):
    summary=[]
    for length in SCALES:
        for m in NAMES:
            selected=[r for r in rows if r['history_s']==length and r['method']==m]
            summary.append({'history_s':length,'method':m,'method_name':NAMES[m],'windows':len(selected),'mean_ADE_10s_m':float(np.mean([r['ADE_10s_m'] for r in selected])),'mean_angle_RMSE_10s_deg':float(np.mean([r['angle_RMSE_10s_deg'] for r in selected]))})
    csvwrite(OUT/'summary.csv',summary)
    lines=['# 同一轨迹的历史长度小规模验证','',
    '选取上一轮图中的测试任务0—5，每类一条。将同一仿真世界延长，检查原始物理轨迹前缀及参考路线完全一致，并保留原时间段的量测噪声。预测时刻固定为240秒与300秒，各自比较12、30、60、120、240秒历史；未来10秒真值完全相同，0.5秒间隔的观测在各历史窗口中严格嵌套。240秒历史是481次观测，12秒是25次；本实验衡量历史积累价值，不是等总观测成本实验。',
    '', '运动模型、空间路径模型、时间周期模型均按相同历史重新初始化并顺序更新。学习融合使用已有冻结门控，不重训；增加固定各占一半融合，区分专家与权重的作用。历史期间每次观测均真实发布预测并积累兑现误差。Chronos、误差区域头、通信调度不在这一轮运行。',
    '', '图中每行显示完整历史和记忆地图、未来轨迹预测、三维误差。每张图同列使用相同坐标范围。类别为生成机制，不保证所选窗口发生停留或偏离。','',
    '## 六条轨迹、两个窗口的平均三维预测误差（米）','',
    '| 历史秒数 | 运动模型 | 空间路径 | 时间周期 | 学习融合 | 固定各半融合 |','|---:|---:|---:|---:|---:|---:|']
    for length in SCALES:
        vals=[next(x['mean_ADE_10s_m'] for x in summary if x['history_s']==length and x['method']==m) for m in NAMES]
        lines.append('| '+str(length)+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','## 地图诊断','', '| 历史秒数 | 有闭合片段的窗口数/12 | 未来真值到记忆路径平均距离（米） | 10秒路线支持度 | 10秒门控路线权重 |','|---:|---:|---:|---:|---:|']
    for length in SCALES:
        ds=[a['diagnostics'] for a in records if a['diagnostics']['history_s']==length]
        lines.append(f'| {length} | {sum(d["closed_fragments"]>0 for d in ds)}/12 | {np.mean([d["future_to_map_mean_m"] for d in ds]):.3f} | {np.mean([d["support_10s"] for d in ds]):.3f} | {np.mean([d["gate_route_weight_10s"] for d in ds]):.3f} |')
    lines+=['','未来真值到记忆路径距离是事后几何诊断，不输入预测器；距离小不代表进度预测正确。闭合片段数也不能单独代表可用记忆，因为开放路线片段和连接也可以复用。支持度表示预测点落在存储路径上的权重，不等于预测正确率。门控仍为短历史数据训练的原模型，因此本实验同时暴露其迁移到长历史的行为，不把门控表现等同于长期路线信息的上限。','',
    '数据文件：metrics.csv保存300行逐方法结果，summary.csv保存总体均值，records.pkl保存完整历史、地图及预测，world_*.npz保存延长的物理轨迹和观测，manifest.json保存实验定义与核对记录。原始数据、模型及旧结果不修改。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render-only',action='store_true');args=p.parse_args()
    if args.render_only:
        with (OUT/'records.pkl').open('rb') as f:records=pickle.load(f)
        with (OUT/'metrics.csv').open(encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        for row in rows:
            for k in ['history_s','ADE_10s_m','angle_RMSE_10s_deg']:row[k]=float(row[k])
    else:records,rows=run()
    plot(records,rows);report(records,rows)
    print('COMPLETE',OUT,flush=True)
