import sys,csv,gzip,pickle,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
B=Path('/data/chen');A=B/'routeclock_archives/paired600'
sys.path[:0]=[str(A),str(A/'tools')]
from ch6_cache import CACHE,version
from routeclock.predictors import apply_gate
cfg=json.loads((A/'configs/research.json').read_text(encoding='utf-8-sig'))
del cfg['simulation']['sensing_budget'],cfg['simulation']['observation_cost_s']
V=version()
with (A/'runs/paired600/models/frozen/bundle.pkl').open('rb') as f:gate=pickle.load(f)['gates']['RC']
def score(w):
    key=['chapter6/test',w,cfg]
    def read(kind,k):
        token=hashlib.sha256(json.dumps([V,k],sort_keys=True).encode()).hexdigest()
        with gzip.open(CACHE/f'{kind}-{token}.pkl.gz','rb') as f:return pickle.load(f)
    world=read('world',key);trs=read('warm',[key,.0031,.05,False,None])
    assert len(trs)==16
    raw=json.loads((B/'routeclock_runs/fixed_history_20261119/boundary'/f'{w:02}.json').read_text())
    fingerprint=hashlib.sha256();rows=[]
    for i,tr in enumerate(trs):
        h=tr.history
        fingerprint.update(np.asarray([len(h)],dtype='<i4').tobytes())
        for o in h:
            for v in (o.t,o.raw,o.position,o.covariance,o.radial_velocity,o.radial_std):fingerprint.update(np.asarray(v,dtype='<f8').tobytes())
        p=tr.snapshots[-1]; apply_gate(p,gate,tr.features(p,'P'),'RC','P')
        for dt in (0,1,3,5,10):
            y,_=world.truth(i,p.origin+dt)
            rows.append([w,i,len(h),p.origin,dt,float(np.linalg.norm(p.at('RC',dt)-y))])
    assert fingerprint.hexdigest()==next(r['history_sha256'] for r in raw if r['cost_ms']==3.1 and r['budget_pct']==5)
    return rows
if __name__=='__main__':
    dest=B/'refine-logs/RC_Chronos2_差距定位/跨版本归因/11月19日固定预热_RC端点位置误差.csv'
    with ProcessPoolExecutor(4) as ex: results=list(ex.map(score,range(30)))
    assert len(results)==30
    with dest.open('w',encoding='utf-8-sig') as fp:
        wr=csv.writer(fp);wr.writerow(['world','uid','warm_observations','forecast_origin_s','horizon_s','position_error_m'])
        for batch in results:wr.writerows(batch)
    rows=np.concatenate(results)
    for dt in (0,1,3,5,10):
        vals=rows[rows[:,4]==dt,5];print(dt,round(vals.mean(),3),np.round(np.quantile(vals,[.5,.9]),3))
