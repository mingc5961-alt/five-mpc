"""Paired descriptive analysis for the long-history service experiment."""
import csv
import json
import pickle
from long_history_data import OUT, CENTERS, KINDS, HORIZONS
from long_history_loop import interpolated, gain_loss
from routeclock.geometry import angles, angle_error
import numpy as np


def read_csv(name):
    with (OUT/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def total(rows, field):
    return sum(float(r[field]) for r in rows)


def mean(rows, field):
    return total(rows,field)/len(rows)


def ratio(a,b):
    return a/b if b else float('nan')


def main():
    loops=read_csv('closed_loop.csv')
    anchors=read_csv('common_anchors.csv')
    assert len(loops)==24 and len(anchors)==72
    table={}
    names={'I':'自适应IMM','P':'空间路径＋进度','envelope':'经验误差包络','lightgbm':'LightGBM','tabpfn':'TabPFN-2.5'}
    lines=['# 长历史下的波束复用与通信闭环','',
           '每种方法使用相同的4个三机场景。每机先按1秒间隔积累180次观测，测试阶段每场景18秒，感知预算5%，每次观测3.1毫秒。每类行为有两条独立测试路线。全部6种组合采用同一阵列方向图、3 dB门限、码率选择和共享调度器。训练、校准和测试分别有24、24和12条独立航线，逐条核对60个几何哈希。建图成本相同，表中吞吐和感知仅计服务期。','',
           '两个位置中心均用同一组可观测诊断特征来预测误差，因此对比隔离的是采用路线位置预测中心的增量价值。预测中心独立训练各自的误差估计器。所有头使用同一批训练和校准观测，每条航线一个锚点、8个时长、两个角度轴。经验包络取训练样本分时长及方向的中位数。LightGBM固定500棵树、15叶、最小叶样本数20，未做测试集选型。TabPFN使用本地真实2.5回归权重、四个估计器，384行上下文。三个头分别用24条校准航线的联合最大误差比分数取第23顺位，名义覆盖90%。','',
           '|位置预测|误差模型|起点32×32可选最长间隔(s)|实际窄束平均保持(s)|窄束占机时(%)|服务期平均观测次数|感知占时(%)|持有段超3 dB次数/段数|净吞吐(Mbps)|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for center in CENTERS:
        for head in KINDS:
            sub=[r for r in loops if r['center']==center and r['head']==head]
            aa=[r for r in anchors if r['center']==center and r['head']==head]
            assert len({r['mission'] for r in sub})==4 and len(aa)==12
            with (OUT/f'head_{center}_{head}.pkl').open('rb') as f:
                model=pickle.load(f)
            d=dict(net=mean(sub,'net_mbps'),cert=mean(aa,'narrow_cert_s'),
                   hold=ratio(total(sub,'narrow_s'),total(sub,'narrow_runs')),share=mean(sub,'narrow_share_pct'),
                   senses=mean(sub,'senses'),sensepct=mean(sub,'sensing_pct'),failed=int(total(sub,'held_failed')),
                   segments=int(total(sub,'held_total')),narrow_fail=int(total(sub,'narrow_failed')),
                   narrow_segments=int(total(sub,'narrow_runs')),covered=int(total(aa,'covered')),
                   fail_air_pct=100*total(sub,'data_bad_s')/total(sub,'data_valid_s'),outage_pct=mean(sub,'outage_pct'),
                   nocover_s=total(sub,'unavailable_s'),factor=float(model.factor),ADE=mean(aa,'ADE_m'),
                   angle=mean(aa,'angle_deg'),unsafe=int(total(aa,'unsafe_narrow_certificate')),
                   issued_narrow=sum(float(r['narrow_cert_s'])>0 for r in aa))
            table[(center,head)]=d
            lines.append(f'|{names[center]}|{names[head]}|{d["cert"]:.2f}|{d["hold"]:.3f}|{d["share"]:.1f}|{d["senses"]:.2f}|{d["sensepct"]:.2f}|{d["failed"]}/{d["segments"]}|{d["net"]:.2f}|')
    lines+=['','起点期限在同一位置预测时刻比较，含无法给32×32提供任何候选的零值。实际保持指同一架机从采用某次预测及32×32子阵到刷新该机预测或改变子阵的连续壁钟时长，发送间隙也包含在内，首尾按18秒服务边界截断。持有段失效为段内20毫秒网格出现任意一次阵列损失超过3 dB。另以实际传输事件的时间计算失效占比。无可用动作的时间另记。','',
            '|位置预测|误差模型|校准倍数|起点联合覆盖/12|窄束证书提前失效/已发|窄束持有段失效/段数|实际传输时间超3 dB(%)|SNR中断占比(%)|不可用累计秒|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for (center,head),d in table.items():
        lines.append(f'|{names[center]}|{names[head]}|{d["factor"]:.2f}|{d["covered"]}/12|{d["unsafe"]}/{d["issued_narrow"]}|{d["narrow_fail"]}/{d["narrow_segments"]}|{d["fail_air_pct"]:.3f}|{d["outage_pct"]:.3f}|{d["nocover_s"]:.3f}|')
    paired=[]
    lines+=['','## 配对收益分解','',
            '每场景的净吞吐等于有效数据时间占比乘以该时间内的平均传输速率。用两种方法的均值作对称分解，得到时间贡献与速率贡献，两者之和精确等于净吞吐变化。有效时间包含感知、切换、不可用动作等开销的影响。速率项包括束宽、对准误差、码率和分时资源分配。','',
            '|误差模型|路线减IMM净吞吐(Mbps)|时间贡献(Mbps)|速率贡献(Mbps)|4个场景配对吞吐差(Mbps)|持有段失效率差(百分点)|','|---|---:|---:|---:|---|---:|']
    for head in KINDS:
        ds=[]
        for mission in range(4):
            i,p=[next(r for r in loops if r['mission']==str(mission) and r['center']==c and r['head']==head) for c in CENTERS]
            ai,ap=[float(r['data_valid_s'])/18 for r in [i,p]]
            ri,rp=[float(r['conditional_mbps']) for r in [i,p]]
            time_effect=(ap-ai)*(rp+ri)/2
            rate_effect=(rp-ri)*(ap+ai)/2
            diff=float(p['net_mbps'])-float(i['net_mbps'])
            np.testing.assert_allclose(time_effect+rate_effect,diff,atol=1e-8)
            ds.append([diff,time_effect,rate_effect])
            paired.append(dict(head=head,mission=mission,delta_net=diff,time_effect=time_effect,rate_effect=rate_effect))
        i,p=[table[(c,head)] for c in CENTERS]
        riskdiff=100*(p['failed']/p['segments']-i['failed']/i['segments'])
        lines.append('|{}|{:.2f}|{:.2f}|{:.2f}|{}|{:+.2f}|'.format(names[head],*np.mean(ds,axis=0),', '.join(f'{x[0]:+.2f}' for x in ds),riskdiff))
    lines+=['','配对比较的独立单位为四个网络任务。预设描述性判断以持有段失效率相差2个百分点以内为相近，同时检查窄束子集和不可用时长。该尺度用于解释此次观测值，四个任务无法统计证明风险等效。','',
            '原始结果见 common_anchors.csv、closed_loop.csv 和各 loop 目录的 choices.json、events.json、predictions.json。每次运行均重算有效比特，核对感知、切换、数据时间之和。']
    truth_rows=[]
    for path in sorted(OUT.glob('network_*.pkl')):
        with path.open('rb') as f:
            samples=pickle.load(f)[0]
        for sample in samples:
            for center in CENTERS:
                pred=interpolated(sample['package'].centers[center],HORIZONS)
                truth=sample['truth']
                loss=gain_loss(angles(pred),angles(truth),(32,32))
                ang=np.rad2deg(angle_error(angles(pred),angles(truth)))
                row={'task':sample['task'],'center':center,'oracle_s':max([h for h in [.5,1,2,3,5,8,10] if np.all(loss[HORIZONS<=h]<=3)] or [0])}
                for h in [1,2,5,10]:
                    row[f'angle_{h}s_deg']=float(np.sqrt(np.sum(ang[HORIZONS<=h]**2,axis=1).mean()))
                truth_rows.append(row)
    with (OUT/'truth_diagnostic.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(truth_rows[0]));writer.writeheader();writer.writerows(truth_rows)
    lines+=['','## 共同起点的事后上限与实际调度','',
            '以20毫秒真值检查32×32子阵的实际阵列损失。取从起点到候选时长始终低于或等于3 dB的最大档位。该量只用于分析。','',
            '|预测器|事后可用间隔(s)|1秒角RMSE(度)|2秒|5秒|10秒|','|---|---:|---:|---:|---:|---:|']
    for center in CENTERS:
        ss=[r for r in truth_rows if r['center']==center]
        lines.append('|'+names[center]+'|'+'|'.join(f'{mean(ss,k):.3f}' for k in ['oracle_s','angle_1s_deg','angle_2s_deg','angle_5s_deg','angle_10s_deg'])+'|')
    lines+=['','再按真实服务阶段的预测发布统计，针对发布当刻选择32×32的记录，比较该次预测的最长可行间隔和调度选择的间隔。以下数据受各方法在线观测时序影响。','',
            '|预测器|误差模型|窄束记录数|可行最长间隔均值(s)|下发间隔均值(s)|','|---|---|---:|---:|---:|']
    for center in CENTERS:
        for kind in KINDS:
            pairs=[]
            for m in range(4):
                folder=OUT/f'loop_{center}_{kind}_{m}'
                events=json.loads((folder/'events.json').read_text())
                forecasts=json.loads((folder/'predictions.json').read_text())
                lookup={(p['uav'],p['origin']):p for p in forecasts}
                for e in events:
                    if e['type']=='observation' and e['pair']==[32,32]:
                        p=lookup[e['uav'],e['origin']]
                        pairs.append((max(a['interval'] for a in p['frontier'] if a['pair']==[32,32]),e['interval']))
            caps,chosen=np.mean(pairs,axis=0)
            lines.append(f'|{names[center]}|{names[kind]}|{len(pairs)}|{caps:.3f}|{chosen:.3f}|')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({f'{k[0]}_{k[1]}':v for k,v in table.items()},indent=2),encoding='utf-8')
    with (OUT/'paired.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)


if __name__=='__main__':
    main()
