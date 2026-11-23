"""Reconcile development receipts and publish auditable world-paired results."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

out = Path(__file__).resolve().parent
root = Path('/data/chen/routeclock_runs/confirm_0db')
scope = json.loads((root/'scope.json').read_text())
ns = scope['analysis']['development_namespace']
caps = {'T':3., 'M':4.5, 'L':9.}
classic = {'F':'F 固定策略','I':'IMM 原管','RC':'RC 原管','RCq90':'RC 分位管'}
models = ('Chronos-2','VECTOR','RAFT')
selected = {m:json.loads((root/'tubes'/m/'selection.json').read_text())['tube'] for m in models}
def label(model,tube):
    if model in classic:
        return classic[model]
    description = {'lightgbm':'分位管','fixed':'固定管','native':'原生管'}[tube]
    return f'{model} {description}'

receipts = {}
replays = []
for m in models:
    checks = json.loads((out/'先进模型硬件复核'/m/'预热复核.json').read_text())
    assert len(checks)==60 and len({(w,b,t) for w,b,t,match in checks})==60
    for w,b,t,match in checks:
        alternate=out/'先进模型硬件复核'/m/f'{w:03}_{b}_{t}.json'
        assert alternate.is_file() == (not match)

for p in sorted((root/'results/development').glob('*/*/*/complete.json')) + sorted((out/'baseline_cells').glob('*.json')):
    rec = json.loads(p.read_text())
    assert rec['namespace']==ns
    m,b,w = rec['model'],rec['budget'],rec['world']
    if m in models:
        alternate = out/'先进模型硬件复核'/m/(p.parent.name+'.json')
        if alternate.exists():
            saved = rec
            rec = json.loads(alternate.read_text())
            assert all(rec[k]==saved[k] for k in ('model','world','budget','tube','namespace','seed','center_sha256'))
            replays.append((m,w,b,rec['tube'],saved['result'],rec['result']))
            p=alternate
    val = rec['result']
    if m in models:
        assert rec['seed']==7 and rec['mcs_margin_db']==0 and rec['risk_scale']==1 and rec['max_interval_s']==10
    elif m!='RCq90':
        assert val['mcs_margin_db']==0 and val['risk_scale']==1 and val['max_interval']==10
        assert abs(val['config']['sensing_budget']*180-caps[b])<1e-10
    tube = rec.get('tube','default')
    key = label(m,tube),b,w
    assert key not in receipts
    air,sw,usable,no_action = val['ledger_s']
    g = val['G_mbps']
    risk,unavailable = val['R']*100,val['unavailable_fraction']*100
    assert len(val['bits'])==16 and val['probes_per_user']==9000
    assert 0<=unavailable<=risk<=100
    assert abs(air+sw+usable+no_action-180)<1e-6 and abs(sum(val['bits'])/180e6-g)<1e-7
    assert air<=caps[b]+.0031+1e-8 and abs(air-val['updates']*.0031)<1e-7
    receipts[key]=(np.array([g,risk,air,100*air/caps[b],unavailable,risk-unavailable,val['narrow_good_data_s'],val['updates']]), val['initial_center_sha256'],str(p))
main = [classic['RC'],classic['RCq90'],*[label(m,'lightgbm') for m in models],classic['I'],classic['F']]
other = [label(m,selected[m]) for m in models]
assert len(receipts)==300 and set(receipts)=={(m,b,w) for m in main+other for b in caps for w in range(10)}
for w in range(10):
    for group in [[classic['RC'],classic['RCq90']]]+[[label(m,'lightgbm'),label(m,selected[m])] for m in models]:
        assert len({receipts[m,b,w][1] for m in group for b in caps})==1

book=Workbook()
overview=book.active
overview.title='口径'
notes=[
    ['项目','说明'],
    ['比较对象','七种配置。F 固定动作；IMM、主线 RC 使用各自归档误差头；RC、Chronos-2、VECTOR、RAFT 另分别接同规格外部 LightGBM 分位管。另列三先进模型择优配管。'],
    ['世界和随机种子',f'{ns}；世界 0–9；每格 10 个世界等权。三个先进模型的冻结中心及外接管使用训练种子 7。使用同一批物理轨迹、噪声和阴影。'],
    ['配对方式','对同一世界、同一预算配对，随后在世界间求均值。更换误差管时核对预热终点中心指纹相同；服务中调度会改变后续观测时刻。'],
    ['感知预算','三档分别为 180 秒服务窗的 1.667%、2.5%、5%，即 3、4.5、9 秒，16 用户共享。单次观测 3.1 ms。预算使用率以各档秒上限为分母。'],
    ['公共条件','180 秒统一预热，每用户约每秒一次实测，随后计分 180 秒。MCS 余量 0 dB，风险尺度 1。吞吐按总成功比特数除以 180 秒，为全系统净 Mbps。'],
    ['调度','自适应法均以同一通信价值规则在预算内选择阵列和更新期限，最长 10 秒，过期即失效；F 以 IMM 预测中心、8×32 阵列、3 秒周期运行，也受感知预算约束。'],
    ['风险','每 20 ms 检查 16 用户，每场共 144000 个探针。无有效动作或有效动作几何损失超过 3 dB 均计入风险。风险两部分以同一组全部探针为分母。'],
    ['外接误差管','各中心分别用相同 240 条训练轨迹、相同外接可观测特征拟合各自的 0.90 分位 LightGBM（130 棵树、15 叶），400 条校准轨迹按期限校准；各自学习的树与校准因子独立。'],
    ['闭环窄波束','32×32 阵列在实际 TDMA 数据段上几何损失不超过 3 dB 的累计秒数，每个 180 秒服务窗均值。'],
    ['配对差及区间','差值方向为左减右。区间为成组世界配对 bootstrap 的 2.5% 与 97.5% 分位数，5000 次、随机种子 271828。十个开发世界的描述性区间，未经多重比较校正。'],
    ['硬件复核',f'用本机冻结权重与相同世界复核全部 180 个先进模型格的初始中心指纹；其中 {len(replays)} 格按本机重放，表中取复核后的结果。独立工作表同时保留原值。'],
    ['正式确认集','原 40 个确认世界仅跑各先进模型择优管：Chronos-2 原生管、VECTOR 与 RAFT 固定管。此处逐世界同管对照来自开发集。'],
]
for line in notes: overview.append(line)
cols=['净吞吐 (Mbps)','总风险 (%)','感知实用 (s)','预算使用率 (%)','无动作 (%)','几何超限 (%)','32×32 有效数据 (s)','感知更新次数']
def matrix(m,b):
    return np.stack([receipts[m,b,w][0] for w in range(10)])
def add_mean_sheet(name,names):
    sh=book.create_sheet(name)
    sh.append(['方法','预算','预算上限 (s)','世界数',*cols])
    for b,cap in caps.items():
        for m in names:
            sh.append([m,b,cap,10,*matrix(m,b).mean(0)])
add_mean_sheet('同预算七配置',main)
add_mean_sheet('先进模型换管对照',[v for m in models for v in (label(m,'lightgbm'),label(m,selected[m]))])
sh=book.create_sheet('逐世界配对差')
sh.append(['左方法','右方法','预算','净吞吐差 (Mbps)','区间下限 (Mbps)','区间上限 (Mbps)','风险差 (百分点)','区间下限 (百分点)','区间上限 (百分点)','感知用时差 (s)','窄波束有效秒数差 (s)','左方吞吐更高世界数'])
idx=np.random.default_rng(271828).integers(0,10,(5000,10))
pairs=[(label(m,'lightgbm'),target) for m in models for target in [classic['RC'],classic['RCq90'],label(m,selected[m])]]
pairs.extend([(classic['RCq90'],classic['RC']),(classic['RC'],classic['I']),(classic['RC'],classic['F'])])
for l,r in pairs:
    for b in caps:
        d=matrix(l,b)-matrix(r,b)
        intervals=np.quantile(d[idx,:2].mean(1),[.025,.975],axis=0)
        sh.append([l,r,b,d[:,0].mean(),*intervals[:,0],d[:,1].mean(),*intervals[:,1],d[:,2].mean(),d[:,6].mean(),int((d[:,0]>0).sum())])
sh=book.create_sheet('逐世界结果')
sh.append(['方法','预算','世界','上限 (s)',*cols,'预热中心 SHA-256','原始回执'])
for m in main+other:
    for b in caps:
        for w in range(10):
            vals,sha,p=receipts[m,b,w]
            sh.append([m,b,w,caps[b],*vals,sha,p])
sh=book.create_sheet('迁移复算')
sh.append(['模型','世界','预算','管','归档净吞吐 (Mbps)','本机净吞吐 (Mbps)','差值 (Mbps)','归档风险 (%)','本机风险 (%)','差值 (百分点)','归档感知 (s)','本机感知 (s)'])
for m,w,b,t,a,z in replays:
    sh.append([m,w,b,t,a['G_mbps'],z['G_mbps'],z['G_mbps']-a['G_mbps'],100*a['R'],100*z['R'],100*(z['R']-a['R']),a['ledger_s'][0],z['ledger_s'][0]])
for sheet in book:
    sheet.freeze_panes='C2' if sheet!=overview else 'A2'
    sheet.auto_filter.ref=sheet.dimensions
    for cell in sheet[1]:
        cell.fill=PatternFill('solid',fgColor='264962')
        cell.font=Font(color='FFFFFF',bold=True)
    for col in sheet.columns:
        width=max([len(str(col[0].value))*1.8]+[len(str(c.value))*1.5 if isinstance(c.value,str) else 10 for c in col[1:]])
        sheet.column_dimensions[get_column_letter(col[0].column)].width=min(94,max(14,width+2))
        for cell in col[1:]:
            if cell.data_type=='n' and isinstance(cell.value,(float,np.floating)):
                cell.number_format='0.000'
    if sheet==overview:
        for row in sheet.iter_rows(min_row=2):
            row[1].alignment=Alignment(wrap_text=True,vertical='center')
            sheet.row_dimensions[row[0].row].height=40
file=out/'同世界吞吐风险与预算.xlsx'
book.save(file)
with file.open('rb') as stream:
    check=load_workbook(stream,read_only=True)
    assert [s.max_row for s in check]==[len(notes),22,19,len(pairs)*3+1,301,len(replays)+1]
    check.close()
for m in main+other:
    print(m,[[round(float(v),3) for v in matrix(m,b).mean(0)] for b in caps])
