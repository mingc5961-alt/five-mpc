"""Recompute the pilot tables and check the saved beam/throughput ledger."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pilot_histories import ROOT,OUT
from routeclock.evaluation import paired_bootstrap
from routeclock.geometry import angles,angle_error
from routeclock.config import dump_json

NAMES={'I':'纯短期运动IMM','P':'纯空间路线＋进度','RC':'路线＋短期运动门控','C':'纯Chronos-2','CR_raw':'Chronos-2＋同一路线','F':'固定通信策略'}
def table(cols,rows):
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    return '\n'.join(lines+['| '+' | '.join(map(str,r))+' |' for r in rows])

def constant_diagnostic():
    weights=np.linspace(0,1,101)[:,None];stats={}
    for split in ['train','validation','cal','network']:
        errors=[]
        for path in sorted(OUT.glob(split+'_*.pkl')):
            with path.open('rb') as f:records=pickle.load(f)[0]
            for a in records:
                for h in (.5,1,2,3,5,8,10):
                    pkg=a['package'];pred=(1-weights)*pkg.at('I',h)+weights*pkg.at('P',h)
                    truth=a['truth'][round(h/.02)]
                    errors.append((np.rad2deg(angle_error(angles(pred),angles(truth)))**2).sum(1))
        e=np.array(errors);averages=e.mean(0);k=averages.argmin()
        stats[split]=dict(rows=len(e),short_better=int((e[:,0]<e[:,-1]).sum()),best_constant_grid_weight=float(weights[k,0]),
                angular_mse_imm=averages[0],angular_mse_route=averages[-1],angular_mse_best=averages[k])
    return stats

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--previous',type=Path)
    args=parser.parse_args();res=OUT/'results'
    fc=pd.read_csv(res/'forecast.csv');dl=pd.read_csv(res/'deadlines.csv')
    cl=pd.read_csv(res/'closed_loop.csv');gate=pd.read_csv(res/'gate_events.csv')
    historical=pd.read_csv(res/'historical_gate_events.csv')
    rsk=pd.read_csv(res/'risk_heads.csv')
    with (res/'bundle.pkl').open('rb') as stream:b=pickle.load(stream)
    assert set(h.mode for h in b['heads'].values())=={'local'}
    assert len(cl)==18 and cl.groupby(['method','action_mode']).size().eq(2).all()
    max_bit_error=weight_error=0.
    for directory in sorted(res.glob('loop_*')):
        get=lambda key:json.loads((directory/f'{key}.json').read_text())
        metrics=get('metrics')
        bits=sum(e.get('delivered_bits',0.) for e in get('events'))
        max_bit_error=max(max_bit_error,abs(bits/18e6-metrics['net_mbps']))
        assert abs(metrics['sensing_s']+metrics['switch_s']+metrics['data_s']-18)<1e-8
        for pred in get('predictions'):
            if metrics['method']=='RC' and 'experts' in pred:
                w=np.asarray(pred['gate_weights']['RC'])[:,None]
                combined=w*np.array(pred['experts']['P'])+(1-w)*np.array(pred['experts']['I'])
                weight_error=max(weight_error,float(np.abs(combined-pred['center']).max()))
    assert max_bit_error<1e-7 and weight_error<1e-8
    matches=0
    if args.previous:
        for path in OUT.glob('*.pkl'):
            split,idx=path.stem.rsplit('_',1)
            if split=='validation':split='train'
            with path.open('rb') as f:records=pickle.load(f)[0]
            with (args.previous/f'{split}_{idx}.pkl').open('rb') as f:old=pickle.load(f)[0]
            for x,y in zip(records,old,strict=True):
                assert x['route_hash']==y['route_hash'];np.testing.assert_equal(x['truth'],y['truth'])
                matches+=1
    fixed=ROOT/'runs/fixed_validation'
    ref=json.loads((fixed/'full_reference_check.json').read_text());winner=json.loads((fixed/'winner.json').read_text())
    stats=fc[fc.horizon==10].groupby('method').mean(numeric_only=True)
    pairs=[]
    for h,g in fc.groupby('horizon'):
        for metric in ('ADE_m','FDE_m','angular_RMSE_deg','angular_p95_deg'):
            pivot=g.pivot(index='task',columns='method',values=metric)
            for other in ('I','P','C','CR_raw'):pairs.append(dict(horizon=h,metric=metric,comparison='RC minus '+other,**paired_bootstrap(pivot.RC,pivot[other])))
    pd.DataFrame(pairs).to_csv(res/'paired_forecast.csv',index=False)
    contrast=[]
    for task,group in cl.groupby('task'):
        main=group[(group.method=='RC')&(group.action_mode=='joint')].iloc[0]
        for _,r in group.iterrows():
            fa=(main.data_s-main.unavailable_s)/main.service_s;fb=(r.data_s-r.unavailable_s)/r.service_s
            if min(fa,fb)<=0:continue
            ra=main.net_mbps/fa;rb=r.net_mbps/fb
            contrast.append(dict(task=task,method=r.method,mode=r.action_mode,net=main.net_mbps-r.net_mbps,
                              time=(fa-fb)*(ra+rb)/2,rate=(ra-rb)*(fa+fb)/2))
    pd.DataFrame(contrast).to_csv(res/'communication_decomposition.csv',index=False)
    sections=['# 租卡前的五模型验证结果','## 样本与运行范围',
       f'12条训练路线、6条验证路线、12条独立校准路线、6条测试路线。六个飞行行为各有一个共同测试起点，共组成2个三机任务。每机180次、约180秒历史，闭环服务期18秒。与旧先导试验逐航线配对复核了{matches}条路线的几何哈希与全部501点未来真值。测试结果限于这两个网络。',
       '使用真实冻结Chronos-2；两版共享位置、观测年龄、径向速度和量测协方差，路线版另读同一条可见路线的预测。主表五个误差头共用局部特征、LightGBM家族及相同校准规则。门控用训练路线的7个时长拟合，按任务三折生成样本外训练预测；误差头所见的未来均来自各自的训练数据。',
       '## 位置与窄束期限',
       table(['预测方法','10秒区间ADE／米','终点误差／米','角RMSE／度','事后最窄束／秒','获准最窄束／秒'],[
       [NAMES[k],f'{r.ADE_m:.3f}',f'{r.FDE_m:.3f}',f'{r.angular_RMSE_deg:.3f}',
        f"{dl[(dl.method==k)&(dl['head']=='common_local')].oracle_s.mean():.3f}",f"{dl[(dl.method==k)&(dl['head']=='common_local')].approved_s.mean():.3f}"] for k,r in stats.iterrows()]),
       '## 门控实际使用了多少路线',
       f'6个共同起点、1/2/5/10秒的24个权重范围为{gate.weight.min():.6f}至{gate.weight.max():.6f}。来自同一航线的历史窗口互相关联，下表仅检查门控反应。偏离按真实模拟偏离窗口及离路径超过3米标记；归位为偏离结束后10秒内且距离不超过3米；在路转向按接下来2秒速度方向改变至少15°标记。标签只用于事后分析。',
       table(['5秒预测事件','历史起点数','平均路线权重','路线终点误差／米','IMM终点误差／米','门控终点误差／米'],[[k,len(g),f'{g.weight.mean():.6f}',f'{g.route_m.mean():.2f}',f'{g.imm_m.mean():.2f}',f'{g.gate_m.mean():.2f}'] for k,g in historical[historical.horizon==5].groupby('event')]),
       '## 两个三机任务的闭环',
       table(['方法','决策','净吞吐Mbps','观测次数／场','窄束占机时','窄束平均保持秒¹','持有段超3 dB'],[
       [NAMES[k],mode,f'{g.net_mbps.mean():.2f}',f'{g.observations_service.mean():.1f}',f'{100*g.narrow_share.mean():.1f}%',
        f'{np.average(g.narrow_hold_s.fillna(0),weights=g.narrow_segments):.3f}' if g.narrow_segments.sum() else '无窄束',
        f'{g.hold_failures.sum()}/{g.hold_segments.sum()}'] for (k,mode),g in cl.groupby(['method','action_mode'])]),
       '¹ 跨两场合并窄束持有段，按段数加权。固定参考为16机验证选出的32×32、2秒，在三机先导中直接使用。joint为联合决策，width_only只调束宽，interval_only只调间隔。F使用固定动作与IMM中心；RC的fixed行保留门控位置中心。',
       '## 同一门控中心的误差模型',table(['方法','平均获准秒','获准／6','提前失效／已获准','联合覆盖／6'],[
       [k,f'{g.approved_s.mean():.3f}',int(g.admitted.sum()),f'{int(g.early_failure.sum())}/{int(g.admitted.sum())}',
        str(int(g.covered.sum())) if g.covered.notna().any() else '直接期限无区域'] for k,g in rsk.groupby('head')]),
       'TabPFN使用真实2.5回归权重，输入为与局部LightGBM一致的168条训练查询行、16个字段。全诊断LightGBM与直接期限树共享诊断信息。0次准入的分母为0，不能据此评为低风险。',
       '## 固定基线与独立核验',
       f"16机、180秒预热及180秒服务、5%共享预算。20个独立验证世界 × 6个可行间隔 × 9种束宽全部完成。最优设置为{winner['pair']}、{winner['interval']}秒，验证均值{winner['validation_mbps']:.4f} Mbps。全长度单场非最优候选与常规逐事件模拟器的差值为{ref['absolute_difference_mbps']:.3g} Mbps。18次闭环的最大比特账本偏差为{max_bit_error:.3g} Mbps；按已记录门控权重重建服务期融合位置的最大误差为{weight_error:.3g}米。",
       '## 研究判断',
       '在六条共同测试路线上，门控几乎把全部权重交给路线。它比纯IMM改善了平均位置预测，当前样本没有显示可辨别的专家切换增益。纯Chronos的平均位置误差最小；为Chronos加入路线后，该项指标反而变差。离线最窄束事后潜力和在线实际吞吐量还取决于误差校准、飞行方向与调度。以正式400条测试航线、30个配对网络检验这些现象的可重复性。']
    (ROOT/'LOCAL_RESULTS.md').write_text('\n\n'.join(sections)+'\n',encoding='utf8')
    diagnostic=res/'gate_constant_diagnostic.json'
    dump_json(diagnostic,constant_diagnostic())
    if diagnostic.exists():
        stats=json.loads(diagnostic.read_text())
        note=('\n训练集84个“航线起点×预测时长”中，IMM在25个查询上的角误差小于路线专家。'
              '在0至1、步长0.01的固定路线权重网格上，训练集与验证集的平均角平方误差都在权重1处最低。'
              f"训练集纯路线的平均平方角误差为{stats['train']['angular_mse_route']:.3f}平方度，纯IMM为{stats['train']['angular_mse_imm']:.3f}平方度。"
              '整体损失偏向路线与门控饱和相符；哪些可见证据能识别短期专家占优时刻，仍需在扩大训练航线后验证。'
              '在本轮两个闭环中，只调间隔与门控联调的净吞吐、窄束占比及失效计数也相同。相对纯路线约7.74 Mbps的差值不足以归因于专家切换。\n')
        with (ROOT/'LOCAL_RESULTS.md').open('a',encoding='utf8') as stream:stream.write(note)
if __name__=='__main__':main()

