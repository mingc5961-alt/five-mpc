"""Audit and publish the real-motion, simulated-radio comparisons and live text report."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from collections import Counter, defaultdict
import numpy as np
from field_worlds import OUT
from history import put

TXT=Path('/data/chen/refine-logs/G1实飞随机航迹与吞吐对照.txt')
SOURCES=('amov','neurobem','combined')
NAMES={'amov':'AMOVFLY','neurobem':'NeuroBEM','combined':'跨来源'}
VIEWS={'mixed':'混合','planned':'全规定路线','random':'全随机'}
ARMS={'P01':'固定时刻 · 截止期选机 + G1','P02':'固定时刻 · AoI-DQN 选机 + G1',
      'Q01':'截止期调度 + G1','Q03':'联合 AoI-DQN + G1','Q02':'截止期调度 + 运动外推',
      'Q04':'联合 AoI-DQN + 运动外推','Q05':'截止期调度 + 固定融合 F*',
      'Q07':'联合 AoI-DQN + RAFT','Q08':'联合 AoI-DQN + Chronos','C_due':'截止期调度 + Chronos'}
PAIRS=[('Q01','Q03'),('P01','P02'),('Q03','Q04'),('Q03','Q07'),('Q08','Q03'),('C_due','Q08'),('C_due','Q01'),('Q01','Q05')]
METRICS=('goodput_mbps','risk_pct','unavailable_pct','pointing_loss_pct','angle_2s_deg','random_user_mbps','user_p05_mbps','sensing_s')

def load(path):return json.loads(path.read_bytes())

def write_table(file,rows):
    with file.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def average(values):
    present=[v for v in values if v is not None]
    return float(np.mean(present)) if present else None

def stats(receipt, indices):
    result=receipt['result'];duration=receipt['duration'];bits=np.asarray(result['bits'],dtype=float)
    rates=bits/(duration*1e6)
    assert abs(sum(rates)-result['G_mbps'])<1e-7
    unavail=result['unavailable_fraction']
    assert 0<=unavail<=result['R']<=1
    return dict(goodput_mbps=result['G_mbps'],risk_pct=100*result['R'],unavailable_pct=100*unavail,
                pointing_loss_pct=100*(result['R']-unavail),angle_2s_deg=average(receipt['angle_2s_per_user']),
                random_user_mbps=average(rates[indices]),user_p05_mbps=float(np.quantile(rates,.05)),sensing_s=result['ledger_s'][0])

def audit_and_group():
    metrics=[];panels={}
    for src in SOURCES:
        spec=load(OUT/(src+'_inputs.json'))
        queue=load(OUT/(src+'_queue.json'))['jobs'];proof=load(OUT/(src+'_done.json'))
        assert proof['checked_endpoints']==len(queue)==len(set(tuple(job) for job in queue))
        for d in spec['panels']:
            tag=d['id'];by=defaultdict(list);rows=[]
            assert len({(f['source'],f['id']) for f in d['flights']})==16
            indices=[i for i,f in enumerate(d['flights']) if f['label']=='random']
            for source,panel,row,seed in queue:
                if panel!=tag:continue
                r=load(OUT/'receipts'/f'{tag}_{row}_{seed}.json')
                assert (r['source'],r['panel'],r['row'],r['seed'])==(src,tag,row,seed)
                assert r['scale']==.75 and len(r['events'])==d['quota'] and r['duration']==d['duration']
                assert abs(r['result']['ledger_s'][0]-.0031*d['quota'])<1e-7
                by[row].append(stats(r,indices));rows.append(r)
            assert len({r['physics'] for r in rows})==1 and set(by)==set(ARMS)
            panels[tag]=dict(source=src,view=d['view'],cohort=d['cohort'],duration=d['duration'],quota=d['quota'],
                             flight_count=16,random_count=len(indices),fingerprint=rows[0]['physics'])
            for row,values in by.items():
                expected=3 if row in ('P02','Q03','Q04','Q07','Q08') else 1
                assert len(values)==expected
                metrics.append(dict(source=src,view=d['view'],panel=tag,row=row,policy_runs=expected,
                                    **{key:average([v[key] for v in values]) for key in METRICS}))
    grouped={}
    for src in SOURCES:
        for view in sorted({v['view'] for v in metrics if v['source']==src}):
            for row in ARMS:
                sample=[v for v in metrics if (v['source'],v['view'],v['row'])==(src,view,row)]
                grouped[src,view,row]=dict(independent_assemblies=len(sample),**{k:average([v[k] for v in sample]) for k in METRICS})
    pairs=[]
    for src,view,_ in (key for key in grouped if key[2]=='Q01'):
        samples=[tag for tag,d in panels.items() if (d['source'],d['view'])==(src,view)]
        for first,second in PAIRS:
            a,b=grouped[src,view,first],grouped[src,view,second]
            deltas=[]
            for tag in samples:
                endpoints={v['row']:v for v in metrics if v['panel']==tag}
                deltas.append(endpoints[first]['goodput_mbps']-endpoints[second]['goodput_mbps'])
            delta=a['goodput_mbps']-b['goodput_mbps']
            pairs.append(dict(source=src,view=view,first=first,second=second,assemblies=len(samples),first_mbps=a['goodput_mbps'],
                    second_mbps=b['goodput_mbps'],gain_mbps=delta,gain_pct=100*delta/b['goodput_mbps'],
                    risk_difference_pp=a['risk_pct']-b['risk_pct'],user_p05_difference_mbps=a['user_p05_mbps']-b['user_p05_mbps'],
                    assembly_deltas_mbps=' | '.join(f'{x:.4f}' for x in deltas)))
    return metrics,panels,grouped,pairs

def report():
    receipts=Counter()
    for f in (OUT/'receipts').glob('*.json'):
        receipts[f.name.split('_',1)[0]]+=1
    timestamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    lines=['G1　随机实飞轨迹、AoI-DQN 与净吞吐对照',f'更新于 {timestamp}', '',
           f'闭环回放　{sum(receipts.values())} / 160 场　（8 个轨迹组合 × 20 个策略与种子设置）']
    for s,expected in zip(SOURCES,(80,40,40)):
        status='全部核验通过' if (OUT/(s+'_done.json')).exists() else '回放与核验中'
        lines.append(f'  {NAMES[s]}　{receipts[s]} / {expected}　{status}')
    lines+=['','每场使用 16 架次各自真实连续的轨迹，同场飞行由异时记录组合。28 GHz、400 MHz 链路及虚拟基站在仿真中构建。',
            '净吞吐 = 模型计算的成功交付比特 / 整场墙钟服务时长。已扣定位、切换、不可用时段及误块损失。',
            'AoI-DQN 使用既有训练权重（种子 17、29、41），相同倍率 0.75、相同定位配额。每个组合先对策略种子取均值。',
            '随机实飞按原采集方标记。几何复核以同向相似航段比例 0.70 为强重复阈值。规定路线可包含多种形状。',
            'AMOVFLY 来源 https://github.com/YujiaoHu/AMOVFLY-Dataset',
            'NeuroBEM 架次来源 https://download.ifi.uzh.ch/rpg/NeuroBEM/Flights.txt',
            'DeepSense 23 官方原包需要授权登录，本批未纳入。https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23']
    geo=OUT/'recurrence_and_angles.json'
    if geo.exists():
        data=load(geo)
        lines+=['','随机实飞的视线角误差（度，越小越好）',
                '  按各航次可用的观测次数与预测时长组合等权汇总。同一 NeuroBEM 随机航点程序重复飞行时，先在程序内取均值。',
                '  观测次数候选为 6、8、12、24、40，预测时长 2、3、5 秒。下表包含首次经过的航段。']
        for s in ('amov','neurobem'):
            for key,label in [('all','全部'),('low_repetition','低重复')]:
                g=data['summary'][s+'_'+key];e=g['angle_deg']
                lines.append(f"  {NAMES[s]} {label}　{g['flights']} 架次 / {g['queries']} 次查询　G1 {e['adaptive']:.3f}　纯运动 {e['motion']:.3f}　固定融合 {e['Fstar']:.3f}　RAFT {e['RAFT']:.3f}　Chronos {e['Chronos']:.3f}")
    done=all((OUT/(s+'_done.json')).exists() for s in SOURCES) and geo.exists()
    if done:
        metrics,panels,grouped,pairs=audit_and_group()
        write_table(OUT/'panel_metrics.csv',metrics);write_table(OUT/'paired_throughput.csv',pairs)
        put(OUT/'summary.json',dict(checked_endpoints=sum(receipts.values()),panels=panels,open_geometry=data['summary'],contrasts=pairs))
        lines+=['','16 架次轨迹组合与实际定位占用']
        for tag,d in panels.items():
            lines.append(f"  {tag}　随机 {d['random_count']} / 16　{d['duration']} 秒　{d['quota']} 次定位　定位占用 {d['quota']*.0031:.4f} 秒")
        lines+=['  AMOVFLY 的两个混合组合使用互不重复的架次。每个混合组合的全规定路线对照保留原规定路线用户并共享随机扰动。',
                '  跨来源全随机组合由 10 架次 AMOVFLY 与 6 架次 NeuroBEM 构成，16 架次的全段与实际评价窗口均通过低重复筛查。',
                '  跨来源组合复用部分前面两来源的记录。NeuroBEM 随机航点飞行均来自同一天。','',
                '各策略的模拟通信结果　总吞吐单位 Mbps，风险为超过 3 dB 几何损失或链路不可用的探针比例。']
        for src,view,_ in (k for k in grouped if k[2]=='Q01'):
            lines+=['',f'{NAMES[src]} · {VIEWS[view]}',
                    '  方法                             总净吞吐    风险(%)    用户吞吐P5    随机用户均值    发出预测后2秒角误差(度)']
            for row,title in ARMS.items():
                v=grouped[src,view,row];ru=v['random_user_mbps'];rand='—' if ru is None else f'{ru:.2f}'
                lines.append(f"  {row:5s} {title}　{v['goodput_mbps']:9.2f}　{v['risk_pct']:7.2f}　{v['user_p05_mbps']:9.2f}　{rand:>8s}　{v['angle_2s_deg']:.3f}")
            lines.append('  配对差值（前者减后者）　　吞吐差 Mbps　相对收益　风险差百分点')
            for p in pairs:
                if (p['source'],p['view'])==(src,view):
                    lines.append(f"    {p['first']:5s} − {p['second']:5s}　{p['gain_mbps']:+9.2f}　{p['gain_pct']:+7.2f}%　{p['risk_difference_pp']:+7.2f}")
        lines+=['','统计口径　同一调度种子的联合 AoI-DQN 各预测器共用完全相同的定位对象和时刻。截止期策略按各自预测决定动作。',
                '  用户吞吐 P5 是 16 个用户各自净吞吐的第 5 百分位数。2 秒误差评估已发出的预测，可包含随后被新定位覆盖的预测。',
                '  不同来源分别汇总。单个组合或两个组合只作描述性比较，不用策略种子、用户或探针扩充独立样本数。',
                f'  逐组合结果　{OUT}/panel_metrics.csv',f'  逐对收益与单组合差值　{OUT}/paired_throughput.csv',
                '','全随机组合中的风险构成（占全部用户探针的百分比）','  方法                           总风险       链路不可用        有效期内指向损失超过 3 dB']
        for row in ('Q01','Q03','C_due','Q08'):
            v=grouped['combined','random',row]
            lines.append(f"  {ARMS[row]}　{v['risk_pct']:.2f}%　{v['unavailable_pct']:.2f}%　{v['pointing_loss_pct']:.2f}%")
        lines+=['','优先验证的三个问题',
                '  1. 取得 DeepSense 23 后按独立飞行切分，比较实际所选波束与当时最佳波束的功率差。具备噪声底、速率与误块标定时再换算估计净吞吐。',
                '  2. 在独立开发数据上为各预测器选择满足同一风险上限的配置，冻结后比较留出航次的总吞吐与最低吞吐用户。另报告不可用与指向损失。',
                '  3. 补多个日期的同源随机实飞，至少连续 180 秒；同源对照采用相同观测时长，按架次和日期留出，分别报告冷启动与稳定服务。']
    tmp=TXT.with_suffix('.writing');tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');tmp.replace(TXT)
    return done

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--wait',action='store_true');opts=cli.parse_args()
    while not report() and opts.wait:time.sleep(5)
