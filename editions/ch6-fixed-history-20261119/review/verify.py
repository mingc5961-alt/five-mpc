"""Recompute selected paired contrasts from the portable review inputs (NumPy + pandas)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
def csv(name):return pd.read_csv(root/'data'/f'{name}.csv')
def show(label,values,n):
    assert values.index.is_unique and len(values)==n and values.notna().all()
    d=values.sort_index().to_numpy(dtype=float)
    rng=np.random.default_rng(271828)
    means=d[rng.integers(0,n,(2000,n))].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    print(f'{label}: n={n}, mean={d.mean():+.3f}, 95% CI [{lo:.3f}, {hi:.3f}]')
def net(df,method,cost,budget):
    rows=df[(df.method==method)&(df.scheduler=='value')&np.isclose(df.cost_ms,cost)&np.isclose(df.budget_pct,budget)]
    return rows.set_index('task').net_mbps
runs=csv('合并逐场指标2080')
assert len(runs)==runs.id.nunique()==2080
prev=runs[(runs.split=='test') & (runs.batch!='固定历史')]
new=runs[runs.batch=='固定历史']
base=prev[(prev.cost_ms==3.1)&(prev.budget_pct==5)]
dense=pd.concat([base,new[~new.fixed_action]])
assert (len(prev),len(new),len(base),len(dense))==(660,580,140,660)
key=['cost_ms','budget_pct','scheduler','method','task']
assert set(map(tuple,prev[key].values))==set(map(tuple,dense[key].values)) and not dense.duplicated(key).any()
def gain(df,cost):return net(df,'RC',cost,5)-net(df,'I',cost,5)
for cost in [3.1,6.2,9.3,12.4]:show(f'dense RC-I at {cost} ms, Mbps',gain(dense,cost),30)
for m in ['I','RC']:show(f'{m}: dense minus original at 12.4 ms, Mbps',net(dense,m,12.4,5)-net(prev,m,12.4,5),30)
slope=lambda df:gain(df,12.4)-gain(df,9.3)
show('12.4 minus 9.3 change of RC-I, dense',slope(dense),30)
show('12.4 minus 9.3 change of RC-I, original',slope(prev),30)
show('interaction: dense change minus original change',slope(dense)-slope(prev),30)
for label,df in [('original',prev),('dense',dense)]:
    show(f'{label} CR_raw-C at 1.667%',net(df,'CR_raw',3.1,5/3)-net(df,'C',3.1,5/3),10)
tight=dense[(dense.method=='F')&(dense.budget_pct<2)]
strict=pd.concat([new[new.fixed_action],tight])
show('common 3 s F, tight minus 5%',net(strict,'F',3.1,5/3)-net(strict,'F',3.1,5),30)
finger=csv('固定历史逐场指纹')
assert len(finger)==581 and (finger.groupby('world').history_sha256.nunique()==1).all()
assert finger.history_sha256.nunique()==30 and (finger.counts==','.join(['180']*8+['179']*8)).all()
bound=csv('服务起点预测逐机')
assert len(bound)==25920 and not bound.duplicated(['world','uid','cost_ms','budget_pct','horizon_after_start','method']).any()
hashes=bound[(bound.cost_ms==3.1)&(bound.budget_pct==5)][['world','history_sha256']].drop_duplicates().set_index('world').history_sha256
assert (finger.world.map(hashes)==finger.history_sha256).all()
r=bound[(bound.method=='RC')&(bound.horizon_after_start==3)&(bound.budget_pct==5)]
angular=r.groupby(['world','cost_ms']).mse.mean().pow(.5).unstack()
show('RC boundary +3s RMSE, dense minus sparse degrees',angular[3.1]-angular[12.4],30)
# Cross-check two raw records against the imported table without relying on /data/chen paths.
for folder in (root/'raw_pair').iterdir():
    raw=json.loads((folder/'metrics.json').read_text())
    df=dense if folder.name=='dense' else prev
    x=df[(df.method=='RC') & (df.cost_ms==12.4) & (df.task==raw['task'])].iloc[0]
    np.testing.assert_allclose([x.net_mbps,x.sensing_pct,x.hold_fail_pct,x.worst10_mbps],
                               [raw['net_mbps'],100*raw['sensing_fraction'],100*raw['hold_failures']/raw['hold_segments'],raw['worst10_mbps']])
print('Sample counts, history fingerprints and raw pair checks passed.')
