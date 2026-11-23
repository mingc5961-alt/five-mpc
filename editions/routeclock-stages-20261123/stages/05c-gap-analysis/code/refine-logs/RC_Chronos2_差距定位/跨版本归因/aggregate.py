"""Aggregate by physical world. World-set differences remain descriptive."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils import get_column_letter

O=Path(__file__).resolve().parent
def read(f):return json.loads(f.read_text())
d=pd.DataFrame([read(f) for f in sorted((O/'cells').glob('*.json'))])
assert len(d)==96 and not d[['scene','world','model','tube','warm','policy','margin']].duplicated().any()
m=pd.DataFrame([read(f) for f in sorted((O.parent/'MCS余量配对').glob('*.json'))])
assert len(m)==16
assert (d.G_mbps>0).all() and d.R.between(0,1).all()
assert all(abs(sum(l)-180)<1e-6 and l[0]<=3.0031+1e-6 for l in d.ledger_s)
assert d.ledger_s.map(lambda l: abs(l[0]/.0031-round(l[0]/.0031))<1e-4).all()
assert (m.updates*.0031<=3.0031+1e-8).all()
m['archive_G_mbps']=[json.load(open(f))['result']['G_mbps'] for f in m.archive_0db]
m['相对归档0dB的复跑偏差']=np.where(m.mcs_margin_db==0,m.G_mbps-m.archive_G_mbps,np.nan)
def arm(model,tube=None,scene='new',warm='dense',policy='strict',margin=0,metric='G_mbps'):
    if tube is None:tube='default' if model in ('RC','C19','I','P') else 'native'
    a=d[(d.model==model)&(d.tube==tube)&(d.scene==scene)&(d.warm==warm)&(d.policy==policy)&(d.margin==margin)].sort_values('world')
    assert len(a)==4,(model,tube,scene,warm,policy,margin)
    return a.set_index('world')[metric]
def gap(**kw):return arm('tuned',**kw)-arm('RC',**kw)
def mcs(margin):
    a=m[m.mcs_margin_db==margin].pivot(index='world',columns='model',values='G_mbps');return a['Chronos-2']-a.RC
legacy=Path('/data/chen/RouteClock_第六章完整实验_20261119/数据表')
exact=pd.DataFrame([read(f) for f in sorted((O/'exact_old').glob('*.json'))])
assert len(exact)==8 and exact.physical_arrays_match.all() and exact.groupby('world').history_hash.nunique().eq(1).all()
exact_gap=exact.pivot(index='world',columns='method',values='modern_G')
older=pd.read_csv(legacy/'合并逐场指标2080.csv')
old=older[(older.batch=='固定历史')&(older.split=='test')&(older.cost_ms==3.1)&(older.budget_pct<2)&(older.scheduler=='value')&older.method.isin(['C','RC'])&older.world.isin([0,2,5,7])].pivot(index='world',columns='method',values='net_mbps')
oldgap=old.C-old.RC
items=[]
def compare(label,before,after):
    assert set(before.index)==set(after.index)
    v=after-before
    items.append(dict(比较=label,世界数=len(v),起点均值=before.mean(),终点均值=after.mean(),平均变化=v.mean(),最小变化=v.min(),最大变化=v.max(),增加世界数=(v>0).sum(),逐世界变化=' / '.join(f'{w}={e:+.2f}' for w,e in v.items())))
for t in ('native','lightgbm'):
    compare('冻结权重下，从旧输入改为新输入，'+t,arm('old',t),arm('frozen',t))
    compare('新输入下，从冻结改为微调，'+t,arm('frozen',t),arm('tuned',t))
for name in ('old','frozen','tuned'):
    compare(name+'，从专用分位树改为原生管',arm(name,'lightgbm'),arm(name,'native'))
compare('旧中心，从第六章归档误差头及乘子改为新分位树及乘子',arm('C19'),arm('old','lightgbm'))
compare('交互：微调后选原生管的收益 vs 冻结时该收益',arm('frozen')-arm('frozen','lightgbm'),arm('tuned')-arm('tuned','lightgbm'))
compare('两模型差，MCS余量0→5 dB',mcs(0),mcs(5))
compare('两模型差，密集→稀疏预热',gap(),gap(warm='sparse'))
compare('两模型差，相同实际动作，累计至所选期限→累计至预测包10秒',gap(),gap(metric='ten_second_G'))
compare('两模型差，放宽过期候选重选，保持严格结算',gap(),gap(policy='reuse'))
compare('两模型差，同时放宽过期候选重选和记账',gap(),gap(policy='reuse',metric='ten_second_G'))
compare('两模型差，规划时按名义→保守SNR选档',gap(),gap(policy='conservative'))
compare('路线模块，IMM→P，各自专用头',arm('I'),arm('P'))
compare('门控模块，P→RC，各自专用头',arm('P'),arm('RC'))
compare('旧世界原版C减RC vs 新执行平台移植原版中心及头，5dB',oldgap,arm('C19',scene='old',margin=5)-arm('RC',scene='old',margin=5))
compare('旧世界原版C减RC vs 本轮完整系统0dB',oldgap,gap(scene='old'))
compare('RC更换同规格分位树与独立校准',arm('RC'),arm('RC','lightgbm'))
compare('同一外接头训练协议，RC→微调Chronos',arm('RC','lightgbm'),arm('tuned','lightgbm'))
compare('相对于RC，旧C移植头→新输入微调原生管，同世界同0dB',arm('C19')-arm('RC'),gap())
compare('完全锁定旧世界、旧预热状态、双方原门控/中心/头/乘子与5dB，只切换整个执行平台',oldgap,exact_gap.C-exact_gap.RC)
compare('共用同一个归档RC尺度头与乘子，只换中心：IMM→路线专家',arm('I','same_RC'),arm('P','same_RC'))
compare('共用同一个归档RC尺度头与乘子，只换中心：路线专家→G1融合',arm('P','same_RC'),arm('RC'))
rc_margin=m[m.model=='RC'].pivot(index='world',columns='mcs_margin_db',values='G_mbps')
compare('11月19日冻结C与当期RC的差值，新四世界，MCS 0→5 dB',arm('C19')-rc_margin[0],arm('C19',margin=5)-rc_margin[5])
factors=pd.DataFrame(items)
a=pd.read_csv(O/'同历史输入与训练.csv')
g=pd.read_csv(O/'门控与专家上限.csv')
assert len(a)==len(g)==12800
def anchors(frame):
    metrics=['position_m','angle_deg']+(['safe32'] if 'safe32' in frame else [])
    return frame.groupby(['model','horizon_s','world'])[metrics].mean().groupby(['model','horizon_s']).mean().reset_index()
means=d.groupby(['scene','model','tube','warm','policy','margin']).agg(n=('world','count'),G=('G_mbps','mean'),risk_pct=('R',lambda r:r.mean()*100),narrow_s=('narrow_good_data_s','mean'),ten_s_G=('ten_second_G','mean'),ten_s_risk_pct=('ten_second_risk',lambda r:r.mean()*100)).reset_index()
assert means.n.eq(4).all()
means.to_csv(O/'闭环配置均值.csv',index=False,encoding='utf-8-sig')
factors.to_csv(O/'配对因素效应.csv',index=False,encoding='utf-8-sig')
with pd.ExcelWriter(O/'跨版本归因证据.xlsx',engine='openpyxl') as xl:
    for sheet,table in [('配对因素效应',factors),('闭环配置均值',means),('96场闭环',d),('16场MCS',m),('输入与权重同历史均值',anchors(a)),('门控事后对照均值',anchors(g)),('11月19日原世界',old.reset_index()),('原状态跨执行器8场',exact),('原版紧档门控与校准',pd.read_csv(legacy/'验证机制配对.csv').query('budget_pct < 2 and metric=="net_mbps"')),('输入逐锚点',a),('门控逐锚点',g)]:
        table.to_excel(xl,index=False,sheet_name=sheet)
        ws=xl.sheets[sheet];ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:
            cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='225772')
        for j,c in enumerate(table.columns,1):
            ws.column_dimensions[get_column_letter(j)].width=min(76,max(14,len(str(c))*1.7+3,max((len(str(s))*(1.6 if any(ord(t)>2000 for t in str(s)) else 1) for s in table[c].head(100)),default=0)+2))
        for line in ws.iter_rows(min_row=2):
            for cell in line:
                if isinstance(cell.value,float):cell.number_format='0.000'
print(means.to_string(index=False));print(factors[['比较','平均变化','最小变化','最大变化']].to_string(index=False))
