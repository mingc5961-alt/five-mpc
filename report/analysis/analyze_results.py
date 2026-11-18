"""Read-only analysis of final RouteClock artifacts; Python 3 + numpy/pandas/matplotlib.
Usage: python analyze_results.py [run_directory] [report_directory]
"""
from pathlib import Path
import sys,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/data/chen/formal_execution/run_20261117_001732')
O=Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).resolve().parents[1]
T=O/'tables'; F=O/'figures';T.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':140,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
COL={'I':'#687787','P':'#d3a128','RC':'#007e87','C':'#7d62a3','CR_raw':'#c77654','F':'#242d3b'}
def save(df,name):
 if isinstance(df.columns,pd.MultiIndex):
  df=df.copy();df.columns=['_'.join(str(v) for v in c if str(v)) for c in df.columns]
 df.to_csv(T/(name+'.csv'),index=False,encoding='utf-8-sig');return df
def ci(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];b=np.random.default_rng(271828).choice(x,(2000,len(x)),replace=True).mean(1)
 return [float(x.mean()),*np.quantile(b,[.025,.975]).tolist(),len(x),int((x>0).sum())]
def paired(d,a,b,metrics,key='mission'):
 x=d[d.policy==a].set_index(key);y=d[d.policy==b].set_index(key);ids=x.index.intersection(y.index);assert len(ids)==len(x)==len(y)
 return [dict(comparison=f'{a} - {b}',metric=m,**dict(zip(['mean_difference','ci_low','ci_high','n','positive_pairs'],ci(x.loc[ids,m]-y.loc[ids,m])))) for m in metrics]
def plot_done(name):plt.tight_layout();plt.savefig(F/(name+'.png'),bbox_inches='tight');plt.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close()
D=pd.read_csv(R/'comparison/all_network_jobs.csv');assert len(D)==645
M=D[(D['head']=='lightgbm')&(D.seed==7)&(D.block=='main')].copy();assert len(M)==180
assert not D.duplicated(['head','seed','block','mission','policy','target_reuse']).any()
metrics=['net_mbps','hold_failure_fraction','sensing_fraction','worst10_mbps','narrow_share','narrow_hold_s','outage_data_fraction','unavailable_s','overdue_requests','observations_service','mean_update_cpu_s','seconds']
main=save(M.groupby('policy')[metrics].mean().reset_index(),'main_means')
rows=[]
for b in ['F','I','P','C','CR_raw']:rows+=paired(M,'RC',b,metrics)
save(pd.DataFrame(rows),'main_paired')
save(D.groupby(['head','seed','block','target_reuse','policy'],dropna=False)[metrics].agg(['mean','std','min','max']).reset_index(),'all_network_descriptive')
# per-route offline summaries; three anchors never treated as independent tasks
Q=pd.read_csv(R/'five/lightgbm_7/forecast.csv');q=Q.groupby(['task','group','method','horizon'],as_index=False)[['ADE_m','FDE_m','angular_RMSE_deg','angular_p95_deg']].mean();assert q.task.nunique()==400
save(q,'offline_per_route');save(q.groupby(['horizon','method']).mean(numeric_only=True).reset_index(),'offline_horizons')
save(q[q.horizon==10].groupby(['group','method']).mean(numeric_only=True).reset_index(),'offline_behavior_10s')
q['protocol']=q.task.str.rsplit('/',n=1).str[-1].astype(int).mod(6).map({0:'0.5s',1:'1s',2:'2s',3:'5s',4:'10s',5:'irregular'})
save(q.groupby(['protocol','horizon','method']).mean(numeric_only=True).reset_index(),'offline_protocol')
rows=[]
for h,g in q.groupby('horizon'):
 for b in ['I','P','C','CR_raw']:
  x=g[g.method=='RC'].set_index('task');y=g[g.method==b].set_index('task')
  for met in ['ADE_m','FDE_m','angular_RMSE_deg','angular_p95_deg']:
   rows.append(dict(horizon=h,comparison='RC - '+b,metric=met,**dict(zip(['mean_difference','ci_low','ci_high','n','positive_pairs'],ci(x[met]-y[met])))))
save(pd.DataFrame(rows),'offline_paired')
G=pd.read_csv(R/'five/lightgbm_7/gate_events.csv');save(G.groupby(['event','horizon'])[['weight','support','I_deg','P_deg','RC_deg','C_deg','CR_raw_deg']].mean().reset_index(),'gate_event_means')
V=pd.read_csv(R/'comparison/offline_validity_task_means.csv');save(V,'offline_validity')
# matched training seeds and matched head replacement
rows=[];sm=[]
for s in [7,11,19]:
 g=D[(D['head']=='lightgbm')&(D.seed==s)&(D.mission<20)&D.block.isin(['main','repeat'])&D.policy.ne('F')]
 assert len(g)==100
 z=g.groupby('policy')[metrics].mean().reset_index();z['seed']=s;sm.append(z)
 for b in ['I','P','C','CR_raw']:
  for v in paired(g,'RC',b,['net_mbps','hold_failure_fraction','sensing_fraction']):v['seed']=s;rows.append(v)
save(pd.concat(sm),'seeds_matched20_means');save(pd.DataFrame(rows),'seeds_matched20_paired')
tab=D[D['head']=='tabpfn']; lgb=M[(M.mission<11)&M.policy.ne('F')];assert len(tab)==len(lgb)==55
rows=[];hm=[]
for p in ['I','P','RC','C','CR_raw']:
 a=tab[tab.policy==p].set_index('mission');b=lgb[lgb.policy==p].set_index('mission')
 for met in metrics:
  rows.append(dict(policy=p,metric=met,tab_mean=a[met].mean(),lgb_mean=b[met].mean(),**dict(zip(['mean_difference','ci_low','ci_high','n','positive_pairs'],ci(a[met]-b[met])))))
 for name,g in [('TabPFN',a),('LightGBM',b)]:hm.append(dict(head=name,policy=p,**g[metrics].mean().to_dict()))
save(pd.DataFrame(rows),'head_matched11_paired');save(pd.DataFrame(hm),'head_matched11_means')
A=pd.concat([D[(D['head']=='lightgbm')&(D.seed==7)&D.block.eq('actions')],M[M.policy=='RC']]);save(A.groupby('policy')[metrics].mean().reset_index(),'actions_means')
rows=[]
for b in ['fixed','width_only','interval_only']:rows+=paired(A,'RC',b,metrics[:8])
save(pd.DataFrame(rows),'actions_paired')
U=D[D.block=='reuse'];save(U.groupby(['target_reuse','policy'])[metrics].mean().reset_index(),'reuse_means')
rows=[]
for r,g in U.groupby('target_reuse'):
 for v in paired(g,'RC','I',metrics[:8]):v['reuse']=r;rows.append(v)
save(pd.DataFrame(rows),'reuse_paired')
# exact symmetric decomposition of throughput = usable data fraction * effective data rate
x=M[M.policy=='RC'].set_index('mission');y=M[M.policy=='I'].set_index('mission')
fx=(x.data_s-x.unavailable_s)/x.service_s;fy=(y.data_s-y.unavailable_s)/y.service_s;rx=x.net_mbps/fx;ry=y.net_mbps/fy
z=pd.DataFrame({'mission':x.index,'total':x.net_mbps-y.net_mbps,'data_time_component':(fx-fy)*(rx+ry)/2,'effective_rate_component':(rx-ry)*(fx+fy)/2}).reset_index(drop=True)
assert np.allclose(z.total,z.data_time_component+z.effective_rate_component)
decomposition_residual=float(np.max(abs(z.total-z.data_time_component-z.effective_rate_component)))
save(z,'throughput_decomposition');save(pd.DataFrame([dict(metric=c,**dict(zip(['mean_difference','ci_low','ci_high','n','positive_pairs'],ci(z[c])))) for c in z if c!='mission']),'decomposition_summary')
N=pd.read_csv(R/'mpc_formal/network_results.csv');assert len(N)==120
save(N.groupby('policy').mean(numeric_only=True).reset_index(),'mpc_means');rows=[]
for b in ['rotation','age_mpc','myopic']:rows+=paired(N,'joint_mpc',b,['Mbps','sensing_pct','over3_pct','worst10_Mbps','jain','terminal_debt_Mbit','observations'])
save(pd.DataFrame(rows),'mpc_paired');save(pd.read_csv(R/'mpc_formal/risk_quality.csv'),'risk_quality');save(pd.read_csv(R/'mpc_formal/tube_quality.csv'),'tube_quality')
E=pd.read_csv(R/'paired/paired_events.csv');assert len(E)==143
save(pd.read_csv(R/'paired/summary.csv'),'refresh_summary');save(pd.read_csv(R/'paired/payback.csv'),'refresh_payback');save(pd.read_csv(R/'paired/prediction_check.csv'),'refresh_prediction_check')
# Explicitly recorded clocks; these are concurrent batch measurements, not controlled latency benchmarks.
save(D.groupby(['head','seed','policy']).agg(n=('seconds','size'),mean_job_s=('seconds','mean'),median_job_s=('seconds','median'),p90_job_s=('seconds',lambda x:x.quantile(.9)),mean_update_s=('mean_update_cpu_s','mean'),max_update_s=('max_update_cpu_s','max')).reset_index(),'runtime_descriptive')
# Detailed replay of mission 0. The RC branch's I expert is evaluated under the SAME observation history.
case=[];ages=[];env=[];gates=[];mcs=[]
for policy,num in [('I',1),('RC',3)]:
 p=R/f'five/network/lightgbm_7/main/{num:03d}'
 events=json.load(open(p/'events.json')); preds=json.load(open(p/'predictions.json')); lookup={(v['uav'],v['origin']):v for v in preds}
 ee=[e for e in events if e['type']=='data' and e['start']>=180 and not e.get('unavailable',False) and 'prediction_age' in e]
 w=np.array([e['data_s'] for e in ee]);a=np.array([e['prediction_age'] for e in ee]);err=np.array([e['actual_angle_error'] for e in ee]);err2=np.mean(err**2,1)
 case.append(dict(policy=policy,age_mean=np.average(a,weights=w),age_le1=np.sum(w[a<=1])/w.sum(),age_le2=np.sum(w[a<=2])/w.sum(),angular_rmse_deg=np.rad2deg(np.sqrt(np.average(err2,weights=w))),past_requested_interval_fraction=sum(e['data_s'] for e in ee if e['start']>e['deadline'])/w.sum(),snr_db=np.average([e['snr_db'] for e in ee],weights=w),predicted_snr_db=np.average([e['predicted_snr_db'] for e in ee],weights=w),bler=np.average([e['bler'] for e in ee],weights=w)))
 for k in sorted(set(e['mcs'] for e in ee)):mcs.append(dict(policy=policy,mcs=k,airtime_fraction=sum(e['data_s'] for e in ee if e['mcs']==k)/w.sum()))
 pp=[v for v in preds if v['origin']>=180]
 for h in [.5,1,2,5,10]:
  env.append(dict(policy=policy,horizon=h,mean_halfwidth_deg=np.mean([np.asarray(v['envelope_rad'])[np.searchsorted(v['h'],h)] for v in pp])*180/np.pi))
  if policy=='RC':gates.append(dict(horizon=h,mean_route_weight=np.mean([v['gate_weights']['RC'][np.searchsorted(v['h'],h)] for v in pp])))
 cf=[]
 if policy=='RC':
  for e in ee:
   v=lookup[(e['uav'],e['prediction_origin'])];xyz=np.array(v['experts']['I']);pos=np.array([np.interp(e['prediction_age'],v['h'],xyz[:,j]) for j in range(3)])
   ang=np.array([np.arctan2(pos[1],pos[0]),np.arctan2(pos[2],np.hypot(*pos[:2]))]);truth=np.array(e['pointing'])+e['actual_angle_error'];diff=(ang-truth+np.pi)%(2*np.pi)-np.pi;cf.append(np.mean(diff**2))
 for lo,hi,label in [(0,1,'0-1s'),(1,2,'1-2s'),(2,10,'2-10s'),(0,10,'all')]:
  mask=(a>lo)&(a<=hi);z=dict(policy=policy,age_bin=label,airtime_share=w[mask].sum()/w.sum(),own_rmse_deg=np.rad2deg(np.sqrt(np.average(err2[mask],weights=w[mask]))))
  if policy=='RC':z['same_history_I_rmse_deg']=np.rad2deg(np.sqrt(np.average(np.array(cf)[mask],weights=w[mask])))
  ages.append(z)
 del events,preds,lookup,ee,pp
save(pd.DataFrame(case),'case0_summary');save(pd.DataFrame(ages),'case0_age');save(pd.DataFrame(env),'case0_tube');save(pd.DataFrame(gates),'case0_gate');save(pd.DataFrame(mcs),'case0_mcs')
# Plots
h=q.groupby(['horizon','method']).angular_RMSE_deg.mean().unstack();fig,ax=plt.subplots(figsize=(8,4))
for p in ['I','P','RC','C','CR_raw']:ax.plot(h.index,h[p],'-o',label=p,color=COL[p])
ax.set(xlabel='Forecast horizon (s)',ylabel='Route-mean angular RMSE (degree)',title='Offline: 400 routes, 3 anchors averaged per route');ax.legend(ncol=5);plot_done('01_offline_horizon')
fig,ax=plt.subplots(figsize=(8,4)); b=q[q.horizon==10].groupby(['group','method']).angular_RMSE_deg.mean().unstack()[['I','P','RC','C','CR_raw']];im=ax.imshow(b,aspect='auto',cmap='YlGnBu');ax.set_xticks(range(5),b.columns);ax.set_yticks(range(len(b)),[x.replace('stable_path_','').replace('_',' ') for x in b.index]);
for i in range(len(b)):
 for j in range(5):ax.text(j,i,f'{b.iloc[i,j]:.2f}',ha='center',va='center',color='white' if b.iloc[i,j]>3 else 'black')
fig.colorbar(im,ax=ax,label='10s angular RMSE (degree)');plot_done('02_behavior')
fig,axs=plt.subplots(1,3,figsize=(12,3.7));order=['F','I','P','RC','C','CR_raw'];mm=main.set_index('policy').loc[order]
for ax,met,scale,title in zip(axs,['net_mbps','hold_failure_fraction','sensing_fraction'],[1,100,100],['Network goodput (Mbps)','Hold segments with >3dB loss (%)','Sensing time (%)']):ax.bar(order,mm[met]*scale,color=[COL[p] for p in order]);ax.set_title(title);ax.tick_params(axis='x',rotation=30)
plot_done('03_main')
pa=pd.DataFrame([r for r in pd.read_csv(T/'main_paired.csv').to_dict('records') if r['metric']=='net_mbps']);fig,ax=plt.subplots(figsize=(7,3.7));ax.errorbar(pa.mean_difference,range(len(pa)),xerr=[pa.mean_difference-pa.ci_low,pa.ci_high-pa.mean_difference],fmt='o',color=COL['RC'],capsize=4);ax.set_yticks(range(len(pa)),pa.comparison);ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Paired goodput difference, Mbps (95% bootstrap CI)',title='30 matched worlds; descriptive intervals');plot_done('04_main_paired')
fig,ax=plt.subplots(figsize=(7,4));aa=A.groupby('policy').mean(numeric_only=True)
for p in aa.index:ax.scatter(aa.loc[p,'hold_failure_fraction']*100,aa.loc[p,'net_mbps'],s=65);ax.annotate(p,(aa.loc[p,'hold_failure_fraction']*100,aa.loc[p,'net_mbps']),xytext=(5,5),textcoords='offset points')
ax.set(xlabel='Hold failure (%)',ylabel='Network goodput (Mbps)',title='Action ablation: all use RC pointing');plot_done('05_actions')
fig,ax=plt.subplots(figsize=(7,4));uu=U.groupby(['target_reuse','policy']).net_mbps.mean().unstack()
for p in ['I','RC']:ax.plot(uu.index*100,uu[p],'-o',color=COL[p],label=p)
ax.set(xlabel='Requested structurally reusable share (%)',ylabel='Network goodput (Mbps)',title='20 paired worlds at each reuse level');ax.legend();plot_done('06_reuse')
fig,axs=plt.subplots(1,2,figsize=(11,4));sm=pd.read_csv(T/'seeds_matched20_means.csv')
for p in ['I','P','RC','C','CR_raw']:g=sm[sm.policy==p];axs[0].plot(g.seed.astype(str),g.net_mbps,'-o',label=p,color=COL[p])
axs[0].set(title='Same 20 worlds: model seeds',ylabel='Goodput (Mbps)',xlabel='Model seed');axs[0].legend(ncol=3)
hm=pd.DataFrame(hm); pv=hm.pivot(index='policy',columns='head',values='net_mbps').loc[['I','P','RC','C','CR_raw']];pv.plot.bar(ax=axs[1],color=['#687787','#007e87']);axs[1].set(title='Same 11 worlds: error-scale head',ylabel='Goodput (Mbps)',xlabel='');axs[1].set_ylim(1100,1250);plot_done('07_robustness')
fig,axs=plt.subplots(1,2,figsize=(10,4)); vv=V[(V.method=='RC')&(V.seed==7)]
for hd,mk in [('lightgbm','o'),('tabpfn','s')]:
 g=vv[vv.error_model==hd]
 for _,r in g.iterrows():
  if np.isfinite(r.mean_halfwidth_deg):axs[0].scatter(r.mean_halfwidth_deg,r.covered*100,marker=mk);axs[0].annotate(hd[:3]+' '+r.variant,(r.mean_halfwidth_deg,r.covered*100),fontsize=8)
  axs[1].scatter(r.approved_s,r.early_failure_given_issued*100,marker=mk);axs[1].annotate(hd[:3]+' '+r.variant,(r.approved_s,r.early_failure_given_issued*100),fontsize=8)
axs[0].axhline(90,ls='--',color='gray');axs[0].set(xlabel='Mean tube halfwidth (degree)',ylabel='Whole-interval coverage (%)');axs[1].set(xlabel='Mean approved duration (s)',ylabel='Early failure among issued (%)');plot_done('08_validity')
fig,axs=plt.subplots(1,3,figsize=(12,4));nm=N.groupby('policy').mean(numeric_only=True).loc[['rotation','myopic','age_mpc','joint_mpc']]
for ax,met,title in zip(axs,['Mbps','sensing_pct','over3_pct'],['Goodput (Mbps)','Sensing (%)','Over-3dB data (%)']):ax.bar(nm.index,nm[met],color=['#687787','#c77654','#7d62a3','#007e87']);ax.set_title(title);ax.tick_params(axis='x',rotation=35)
plot_done('09_mpc')
fig,axs=plt.subplots(1,2,figsize=(10,4));rq=pd.read_csv(T/'risk_quality.csv');tq=pd.read_csv(T/'tube_quality.csv');axs[0].plot(rq.h_s,rq.brier,'-o',label='Individual risk');axs[0].plot(rq.h_s,rq.age_brier,'-o',label='Age only');axs[0].set(xlabel='Horizon (s)',ylabel='Brier score (lower better)');axs[0].legend();axs[1].plot(tq.h_s,tq.tube_covered*100,'-o');axs[1].axhline(90,ls='--',color='gray');axs[1].set(xlabel='Horizon (s)',ylabel='Tube coverage (%)',ylim=(88,100));plot_done('10_mpc_quality')
fig,ax=plt.subplots(figsize=(8,4));rr=pd.read_csv(T/'refresh_summary.csv');hh=['first','matched','locked','1s','3s','5s']
for group,label in [('all','All 143 events'),('scene_0','Scene 0 (75)'),('scene_1','Scene 1 (68)')]:g=rr[rr.group==group].set_index('horizon').loc[hh];ax.plot(hh,g.mean_delta_Mbit,'-o',label=label)
ax.axhline(0,color='gray',ls='--');ax.set(ylabel='Refresh minus wait (Mbit)',title='Paired replay: dependent events from only TWO worlds');ax.legend();plot_done('11_refresh')
fig,axs=plt.subplots(1,3,figsize=(12,4));gg=pd.DataFrame(ages);g=gg[(gg.policy=='RC')&gg.age_bin.ne('all')];ix=np.arange(len(g));axs[0].bar(ix-.16,g.same_history_I_rmse_deg,.32,label='I on RC history');axs[0].bar(ix+.16,g.own_rmse_deg,.32,label='RC');axs[0].set_xticks(ix,g.age_bin);axs[0].set_ylabel('Angular RMSE (degree)');axs[0].legend(fontsize=8)
for p in ['I','RC']:g=pd.DataFrame(env);g=g[g.policy==p];axs[1].plot(g.horizon,g.mean_halfwidth_deg,'-o',label=p,color=COL[p])
axs[1].set(xlabel='Horizon (s)',ylabel='Mean tube halfwidth (degree)');axs[1].legend();g=pd.DataFrame(gates);axs[2].plot(g.horizon,g.mean_route_weight*100,'-o',color=COL['RC']);axs[2].set(xlabel='Horizon (s)',ylabel='Gate weight on route (%)',ylim=(90,100));fig.suptitle('Diagnostic replay of mission 0 only');plot_done('12_case0')
fig,ax=plt.subplots(figsize=(8,4));pr=q[q.horizon==1].groupby(['protocol','method']).angular_RMSE_deg.mean().unstack().loc[['0.5s','1s','2s','5s','10s','irregular']]
for p in ['I','P','RC']:ax.plot(pr.index,pr[p],'-o',label=p,color=COL[p])
ax.set(xlabel='Offline observation protocol (different route subsets)',ylabel='1s angular RMSE (degree)');ax.legend();plot_done('13_protocol')
checks={'five_jobs':len(D),'mpc_jobs':len(N),'offline_routes':q.task.nunique(),'offline_anchor_method_rows':len(Q),'paired_refresh_events':len(E),'main_pairs':30,'matched_repeat_worlds':20,'tabpfn_matched_worlds':11,'bootstrap_replicates':2000,'bootstrap_seed':271828,'confidence_intervals':'paired, route/world cluster bootstrap; not simultaneous; exploratory selection disclosed','decomposition_max_abs_residual':decomposition_residual,'scientific_source_modified_by_report':False}
(O/'analysis/analysis_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print('Analysis finished;',len(list(T.glob('*.csv'))),'tables;',len(list(F.glob('*.png'))),'figures',flush=True)
