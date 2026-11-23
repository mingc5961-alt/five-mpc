"""Readable per-behavior analysis and gallery for the matched history pilot."""
import csv,json,html
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/history_scale_pilot'
with (OUT/'metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==300
for r in rows:
    for k in ['history_s','origin_s','ADE_10s_m','angle_RMSE_10s_deg','future_to_map_mean_m','support_10s','gate_route_weight_10s']:r[k]=float(r[k])
groups=list(dict.fromkeys(r['group'] for r in rows))
def select(group=None,method='P',length=240):
    return [r for r in rows if (group is None or r['group']==group) and r['method']==method and r['history_s']==length]
def avg(group=None,method='P',length=240,key='ADE_10s_m'):
    return float(np.mean([r[key] for r in select(group,method,length)]))
table=['| 行为 | 路径预测：12秒历史 | 路径预测：240秒历史 | 运动模型：240秒历史 | 学习融合：240秒历史 | 固定各半：240秒历史 |','|---|---:|---:|---:|---:|---:|']
for group in groups:
    label=select(group)[0]['behavior'];vals=[avg(group,'P',12),avg(group,'P'),avg(group,'I'),avg(group,'RC'),avg(group,'HALF')]
    table.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
index={(r['task'],r['origin_s'],r['history_s'],r['method']):r for r in rows}
counts={}
for key in ['ADE_10s_m','angle_RMSE_10s_deg']:
    counts[key]={}
    for a,b in [('P','I'),('RC','I'),('HALF','RC')]:
        counts[key][f'{a}_better_than_{b}']=sum(r[key]<index[(r['task'],r['origin_s'],240.,b)][key] for r in select(method=a))
p0=avg(method='P',length=12);p1=avg(method='P');i=avg(method='I');rc=avg(method='RC');half=avg(method='HALF')
improvement=(1-p1/p0)*100
text=['# 历史延长实验：结果解释','',
f'本轮已完成60个历史窗口条件、300组方法结果。六条固定轨迹，每条在240秒和300秒分别预测未来10秒；五种历史长度12/30/60/120/240秒。所有方法参数保持原样，未训练新门控，未修改路线记忆算法。',
'', '## 结果一：12秒历史确实限制了路线记忆，但长历史不是充分条件','',
f'六类、两个预测窗口合计平均，空间路径预测误差从12秒历史的{p0:.3f}米变为240秒历史的{p1:.3f}米，下降{improvement:.1f}%。同样240秒历史下，运动模型为{i:.3f}米、学习融合为{rc:.3f}米、固定各半融合为{half:.3f}米。以下每行是同一条轨迹的两个窗口平均，指标为未来10秒平均三维位置误差，单位米。','',*table,'',
f'240秒历史下，空间路径预测在12个窗口中的{counts["ADE_10s_m"]["P_better_than_I"]}个窗口优于运动模型；学习融合在{counts["ADE_10s_m"]["RC_better_than_I"]}个窗口优于运动模型。按与波束直接相关的角度RMSE计算，对应是{counts["angle_RMSE_10s_deg"]["P_better_than_I"]}个和{counts["angle_RMSE_10s_deg"]["RC_better_than_I"]}个窗口。','',
'## 结果二：当前实现至少有两处需要分开看','',
'1. 路线几何已经被记录，不代表未来预测会沿正确路段继续。稳定路径任务0在第300秒、60秒历史条件下，未来真值到记忆路径平均距离只有约0.49米，但没有形成闭合片段，10秒路线支持度为0，空间路径平均误差约26.80米。该例应先检查闭合/接续和进度映射，而不能归为“历史中没有路线”。',
'2. 原门控未充分保留路线改善。分段变速任务2在第240秒、240秒历史下，空间路径预测平均误差约1.18米，运动模型约7.71米，学习融合约5.86米。现有门控来自短历史训练，本次冻结它是为了观察迁移表现；这些结果表明有必要单独检验长历史训练后的融合，但不证明重训一定解决问题。','',
'## 结果三：随机路线检验了记忆误用','',
'无可复用路径类别必须单独看。它也可能靠近历史经过的位置，但未来不一定沿原路前进。表中的这一行，以及其路线支持度、实际误差，能够检验“落在已有路径上”是否被误当成“预测正确”。本轮不因结果好坏改变类别、参数或样本。','',
'## 下一步优先级','',
'先复核几何覆盖充分但预测仍失败的闭合/接续案例，再用长短历史共同组成训练集，单独比较旧门控、重训门控和固定融合。保持当前六条测试轨迹不进入训练。完成后再进入误差区域和通信收益验证。本轮没有执行这些修改。','',
'## 图与数据','',
'`00_history_length_vs_error.png`是总览；每类有t240和t300两张五行详图。`index.html`可以按类别浏览全部图片。`REPORT.md`说明协议及总体结果；`metrics.csv`保存全部节点指标；`records.pkl`保存历史、地图和预测。',
'', '这轮比较的是已有历史是否有用。240秒历史包含481次观测，12秒包含25次观测；感知成本收益要在后续共同预算的闭环中测量。两个窗口来自同一条轨迹，不能视作两条独立航线。']
(OUT/'ANALYSIS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
diagnostic={'mean_P_ADE_12s':p0,'mean_P_ADE_240s':p1,'mean_I_ADE_240s':i,'mean_RC_ADE_240s':rc,'mean_HALF_ADE_240s':half,'P_improvement_percent':improvement,'window_counts':counts,'angle_means_240s':{m:avg(method=m,key='angle_RMSE_10s_deg') for m in ['I','P','H','RC','HALF']}}
(OUT/'analysis_numbers.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>历史长度实验图集</title><style>body{font:16px "Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;background:#f4f6f8;color:#18232c}img{max-width:100%;background:white}section{margin:40px 0}a{color:#1260a1}p{line-height:1.7}</style><h1>同一轨迹，延长历史后路线记忆有没有用？</h1><p>六条轨迹 × 两个固定预测时刻 × 五种历史长度。每张详图从上到下为12、30、60、120、240秒历史；左列完整历史与地图，中列未来预测，右列三维误差。点击图片可打开原尺寸。</p><p><a href="ANALYSIS.md">结果分析</a> · <a href="metrics.csv">全部数值</a></p><img src="00_history_length_vs_error.png">']
for i,group in enumerate(groups):
    label=select(group)[0]['behavior'];parts.append(f'<section><h2>{html.escape(label)}</h2>')
    for t in [240,300]:
        file=f'{i+1:02d}_{group}_t{t}.png';parts.append(f'<details><summary>第{t}秒预测：展开完整历史与未来轨迹</summary><a href="{file}"><img loading="lazy" src="{file}"></a></details>')
    parts.append('</section>')
parts.append('</html>');(OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(diagnostic,ensure_ascii=False,indent=2))
