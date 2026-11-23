"""Read-only diagnostic on held-out stored forecasts; oracle uses future truth.
Oracle is unattainable online and is NOT a newly trained model or deployable result.
It allows independent mixing at every 0.02 s truth point, more flexible than the actual gate/grid.
"""
from pathlib import Path
import sys,pickle,json
import numpy as np,pandas as pd
sys.path.insert(0,'/data/chen/routeclock_server/routeclock_five_20261115')
from routeclock.geometry import angles,angle_error
from routeclock.predictors import apply_gate
R=Path('/data/chen/formal_execution/run_20261117_001732');O=Path(__file__).resolve().parent
bundle=pickle.load(open(R/'five/lightgbm_7/bundle.pkl','rb'));gate=bundle['gates']['RC'];rows=[];diag=[]
phi=(np.sqrt(5)-1)/2
for ix,path in enumerate(sorted((R/'five_data/test').glob('*.pkl'))):
 for a in pickle.load(open(path,'rb')):
  pkg=a['package'];apply_gate(pkg,gate,a['features']['P'],'RC','P');hs=a['truth_h'];truth=a['truth'];ta=angles(truth)
  pred={k:np.stack([np.interp(hs,pkg.hs,pkg.centers[k][:,j]) for j in range(3)],axis=1) for k in ['I','P','RC','C']}
  def loss(w):
   z=pred['I']+w[:,None]*(pred['P']-pred['I']);return np.mean(np.rad2deg(angle_error(angles(z),ta))**2,axis=1)
  # Coarse global grid brackets potential minima, then vectorized golden search.
  grid=np.linspace(0,1,41);values=np.stack([loss(np.full(len(hs),v)) for v in grid]);idx=values.argmin(0);lo=grid[np.maximum(idx-1,0)];hi=grid[np.minimum(idx+1,len(grid)-1)]
  x1=hi-phi*(hi-lo);x2=lo+phi*(hi-lo);f1=loss(x1);f2=loss(x2)
  for _ in range(28):
   left=f1<f2;hi=np.where(left,x2,hi);lo=np.where(left,lo,x1);x1=hi-phi*(hi-lo);x2=lo+phi*(hi-lo);f1=loss(x1);f2=loss(x2)
  w=(lo+hi)/2;err={k:np.mean(np.rad2deg(angle_error(angles(v),ta))**2,axis=1) for k,v in pred.items()};oracle=np.minimum.reduce([loss(w),err['I'],err['P']]);err['oracle_mix']=oracle;err['oracle_select']=np.minimum(err['I'],err['P'])
  assert np.all(oracle<=err['I']+1e-10) and np.all(oracle<=err['P']+1e-10)
  for h in [.5,1,2,3,5,8,10]:
   sel=hs<=h+1e-8
   for k,e in err.items():rows.append(dict(task=a['task'],group=a['group'],origin=pkg.origin,horizon=h,method=k,angular_RMSE_deg=float(np.sqrt(e[sel].mean()))))
  # Geometric discrepancy on P route and its smoothed/extrapolated center, descriptive only.
  diag.append(dict(task=a['task'],origin=pkg.origin,group=a['group'],event=a['event'],route_weight_1=float(pkg.diagnostics['gate_weights']['RC'][2]),route_weight_10=float(pkg.diagnostics['gate_weights']['RC'][20]),support_10=float(pkg.diagnostics['support'][20]),progress_std_10=float(pkg.diagnostics['progress_std'][20]),match_entropy=float(pkg.diagnostics['memory'].get('match_entropy',0))))
 if ix%100==0:print('Processed routes',ix+1,flush=True)
d=pd.DataFrame(rows);q=d.groupby(['task','group','horizon','method']).angular_RMSE_deg.mean().reset_index();q.to_csv(O/'oracle_per_route.csv',index=False);avg=q.groupby(['horizon','method']).angular_RMSE_deg.mean().unstack();avg.to_csv(O/'oracle_horizon_means.csv');q[q.horizon==10].groupby(['group','method']).angular_RMSE_deg.mean().unstack().to_csv(O/'oracle_behavior_10s.csv');pd.DataFrame(diag).to_csv(O/'anchor_diagnostics.csv',index=False)
actual=pd.read_csv(R/'five/lightgbm_7/forecast.csv').groupby(['horizon','method']).angular_RMSE_deg.mean().unstack()[['I','P','RC','C']];delta=np.max(np.abs(avg[actual.columns].values-actual.values));assert delta<1e-9,delta
rows=[]
for h in [1,2,5,10]:
 pv=q[q.horizon==h].pivot(index='task',columns='method',values='angular_RMSE_deg')
 for first,last in [('RC','C'),('oracle_mix','C'),('RC','oracle_mix')]:
  dif=(pv[first]-pv[last]).values;boot=np.random.default_rng(271828).choice(dif,(2000,len(dif))).mean(1);rows.append(dict(horizon=h,comparison=first+' - '+last,mean=float(dif.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),positive_pairs=int((dif>0).sum()),n=len(dif)))
pd.DataFrame(rows).to_csv(O/'oracle_paired_ci.csv',index=False)
(O/'checks.json').write_text(json.dumps({'routes':q.task.nunique(),'anchors':len(diag),'original_forecast_max_abs_difference':delta,'oracle_uses_future_truth':True,'oracle_is_deployable':False,'scientific_model_or_config_modified':False,'bootstrap_cluster':'route','bootstrap_replicates':2000},indent=2))
print(avg.to_string());print(pd.DataFrame(rows).to_string(index=False))
