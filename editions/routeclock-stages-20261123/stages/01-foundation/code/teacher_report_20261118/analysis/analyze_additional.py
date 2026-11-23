from pathlib import Path
import json,sys
import numpy as np,pandas as pd
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/data/chen/formal_execution/run_20261117_001732');O=Path(__file__).resolve().parents[1];T=O/'tables'
def save(d,n):d.to_csv(T/(n+'.csv'),index=False,encoding='utf-8-sig')
def ci(x):
 x=np.asarray(x,float);a=np.random.default_rng(271828).choice(x,(2000,len(x))).mean(1);return [x.mean(),*np.quantile(a,[.025,.975]),len(x)]
rows=[];audit=[];fair=[]
for p in (R/'five/network/lightgbm_7/main').glob('*/metrics.json'):
 d=json.load(open(p));c=json.load(open(p.with_name('completed.json')));u=np.array(d['uav_mbps']);fair.append(dict(policy=c['policy'],mission=c['mission'],jain=u.sum()**2/(len(u)*sum(u*u)),min_uav_mbps=u.min(),max_uav_mbps=u.max()))
 audit.append(dict(policy=c['policy'],mission=c['mission'],clock_error=abs(d['accounted_s']-d['service_s']),bits_error=abs(sum(x['bits'] for x in d['group_resources'])/d['service_s']/1e6-d['net_mbps']),sensing_excess=d['sensing_fraction']-d['sensing_budget']))
 for uid,g in enumerate(d['group_resources']):rows.append(dict(policy=c['policy'],mission=c['mission'],uid=uid,group=g['group'],uav_mbps=g['bits']/d['service_s']/1e6,sensing_s=g['sensing_s'],assigned_s=g['assigned_s'],unavailable_s=g['unavailable_s']))
x=pd.DataFrame(rows);save(x,'main_per_uav');g=x.groupby(['mission','policy','group']).mean(numeric_only=True).reset_index();save(g.groupby(['policy','group'])[['uav_mbps','sensing_s','assigned_s','unavailable_s']].mean().reset_index(),'main_behavior_means')
rows=[]
for grp,d in g.groupby('group'):
 a=d[d.policy=='RC'].set_index('mission');b=d[d.policy=='I'].set_index('mission')
 for m in ['uav_mbps','sensing_s','assigned_s']:rows.append(dict(group=grp,metric=m,**dict(zip(['difference','ci_low','ci_high','n'],ci(a[m]-b[m])))))
save(pd.DataFrame(rows),'main_behavior_paired');save(pd.DataFrame(fair).groupby('policy').mean(numeric_only=True).reset_index(),'main_fairness');save(pd.DataFrame(audit),'main_accounting_audit')
rows=[]
for bundle in ['lightgbm_7','lightgbm_11','lightgbm_19','tabpfn_7']:
 q=pd.read_csv(R/f'five/{bundle}/forecast.csv');q=q.groupby(['task','method','horizon']).mean(numeric_only=True).reset_index();q=q[(q.method=='RC')&q.horizon.isin([1,2,5,10])];z=q.groupby('horizon').mean(numeric_only=True).reset_index();z['bundle']=bundle;rows.append(z)
save(pd.concat(rows),'offline_seed_head_RC')
rows=[]
for bundle in ['lightgbm_7','tabpfn_7']:
 d=pd.read_csv(R/f'five/{bundle}/deadlines.csv');d['covered']=d.covered.map({True:1,False:0,'True':1,'False':0});g=d.groupby(['task','method','head'])[['covered','issued','early_failure','approved_s','mean_halfwidth_deg']].mean().reset_index()
 for (m,h),v in g.groupby(['method','head']):
  for met in ['covered','issued','early_failure','approved_s']:
   if v[met].notna().all():rows.append(dict(bundle=bundle,method=m,variant=h,metric=met,**dict(zip(['mean','ci_low','ci_high','n'],ci(v[met])))))
save(pd.DataFrame(rows),'offline_validity_cluster_ci')
print('Additional tables complete',len(audit),'world-policy accounting checks')
