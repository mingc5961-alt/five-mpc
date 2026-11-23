from pathlib import Path
import sys,json
import pandas as pd,numpy as np
root=Path('/data/chen/routeclock_runs/confirm_0db')
sys.path.insert(0,str(root))
import summarize_batch as sb
summary=root/'results/summary'
met=sb.METRICS
a=sb.audit(True)
saved=pd.read_csv(summary/'all_cells.csv',dtype={'unit':str},keep_default_na=False)
for col in met: saved[col]=pd.to_numeric(saved[col])
pd.testing.assert_frame_equal(a.reset_index(drop=True),saved,check_dtype=False,rtol=1e-10,atol=1e-10)
win=pd.DataFrame(json.loads((root/'work/flight_windows.json').read_text())).set_index('flight_id')
x=[]
for st in sb.STAGES:
 for p in (root/'results'/st).rglob('complete.json'):
  v=json.loads(p.read_text());r=v['result']; c=v.get('contract',v.get('frozen_config',v)); l=r['ledger_s']; ds=np.asarray(r.get('selected_deadlines',[0]*7))
  x.append(dict(stage=st,unit=str(c.get('world',c.get('flight_id'))),budget=c.get('budget',c.get('condition')),model=c['model'],seed=c.get('seed',0),tube=c.get('tube','default'),obs=r['updates'],sensing=l[0],usable=l[2],idle=l[3],narrow=r['narrow_good_data_s'],active_rate=r['G_mbps']*180/l[2] if l[2] else 0,p10=np.quantile(np.array(r['bits'])/180/1e6,.1),hold=(ds@np.array([.5,1,2,3,5,8,10])/ds.sum() if ds.sum() else np.nan)))
cols=['stage','unit','budget','model','seed','tube']
a=a.merge(pd.DataFrame(x),on=cols,validate='1:1')
a['route']=a.unit.map(win.route)
syn=a[a.stage.isin(['baselines','confirmation'])];fly=a[a.stage=='amovfly'];dev=a[a.stage=='development']
means=lambda d:d.groupby(['budget','model'])[met+['obs','sensing','usable','idle','narrow','active_rate','p10','hold']].mean()
tab=[means(syn),means(fly)]
rng=271828
def pair(data,budget,m,base,axis='model',cluster=False):
 d=data[data.budget==budget]
 mat=d.groupby(['unit',axis])[met].mean()
 diff=mat.xs(m,level=1)-mat.xs(base,level=1)
 rand=np.random.default_rng(rng);n=len(diff);idx=rand.integers(n,size=(5000,n))
 if cluster:
  days=d.drop_duplicates('unit').set_index('unit').loc[diff.index,'session']
  codes=sorted(days.unique()); g=rand.integers(len(codes),size=(5000,len(codes)));w=np.stack([(g==i).sum(axis=1) for i in range(len(codes))],axis=1)[:,[codes.index(i) for i in days]]
  rep=w@diff.to_numpy()/w.sum(1)[:,None]
 else:rep=diff.to_numpy()[idx].mean(1)
 return diff, np.array([diff.mean().values,*np.quantile(rep,[.025,.975],axis=0)]).T
if __name__=='__main__':
 for df,table in zip([syn,fly],tab):
  print(table[['net_mbps','risk_pct','unavailable_pct','geom_pct','obs','usable','narrow','active_rate','p10','hold']].round(2).to_string())
  for b in 'TML':
   for m,ref in [('RC','I'),('Chronos-2','RC')]+([('RC','F')] if df is syn else [('VECTOR','RC'),('RAFT','RC')]):
    d,est=pair(df,b,m,ref); print(b,m,ref,'G/R',est[:2].round(2).tolist(),'wins',int((d.net_mbps>0).sum()))
 print('flight dates',fly.session.nunique(),win.groupby('route')[['warm_lap_equivalents','score_lap_equivalents']].median())
 for g,v in fly.groupby('route'):
  for m,base in [('RC','I'),('Chronos-2','RC')]:
   d,est=pair(v,'T',m,base);print(g,m,base,est[:2],(d.net_mbps>0).sum())
 for m,ref in [('RC','I'),('Chronos-2','RC')]:
  for b in 'TML':
   d,e=pair(fly,b,m,ref,cluster=True);print('date',b,m,ref,e[:2].round(2).tolist())
 print('SEEDS',syn.groupby(['model','seed','budget']).net_mbps.mean().unstack().round(1))
