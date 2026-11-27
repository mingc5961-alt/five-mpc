import itertools
import numpy as np,pandas as pd
from base import *
def workpoint(g,ref,cap=False):
    agg=g.groupby(['a','m'])[['risk','mbps','narrow','sense','unavailable']].mean()
    risks=agg.risk.groupby('a');lo=risks.min().max();hi=risks.max().min()
    target=5. if cap else (float(np.clip(agg.loc[(ref,1.),'risk'],lo,hi)) if lo<=hi else None)
    out={}
    for a,b in agg.groupby(level='a'):
        b=b.droplevel(0);opts=[]
        for k,r in b.iterrows():
            if cap and r.risk<=target or target is not None and abs(r.risk-target)<1e-9:opts.append((r.mbps,k,k,0.))
        if target is not None:
            for x,y in itertools.combinations(b.index,2):
                r,s=b.loc[x],b.loc[y]
                if min(r.risk,s.risk)<=target<=max(r.risk,s.risk) and r.risk!=s.risk:
                    t=(target-r.risk)/(s.risk-r.risk);opts.append((r.mbps*(1-t)+s.mbps*t,x,y,t))
        if opts:out[a]=max(opts)[1:]
    return target,out
def report(t):
    dest=BASE/'results'/t['args']['code'];dest.mkdir(exist_ok=True)
    if t['args']['code']=='core_history':
        from history_report import history_check
        return history_check(t,dest)
    rows=[];pred=[]
    for key in t['deps']:
        if key.startswith('tables__'):
            continue  # A prerequisite report has no per-origin observations.
        p=BASE/'receipts'/(key+'.json');doc=json.loads(p.read_text());a=doc['args'];d=doc['data']
        if 'result' in d:
            r=d['result'];rows.append(dict(**a,risk=100*r['R'],unavailable=100*r['unavailable_fraction'],mbps=r['G_mbps'],narrow=r['narrow_good_data_s'],sense=r['ledger_s'][0],**dict(zip(['closed1','closed2','closed5','closed10'],d['closed_loop_deg']))))
        else:
            with np.load(p.with_suffix('.npz')) as z:
                keys=z['keys'];e=z['errors']
                for j,name in enumerate(z['names']):
                    for label,mask in [('all',keys[:,0]!=''),('revisit',keys[:,2]=='revisit'),('detour',keys[:,2]=='detour'),('early',keys[:,3]=='True'),('random',keys[:,2]=='random'),('reusable',keys[:,2]!='random'),('first',keys[:,2]=='first')]:
                        pred.append(dict(**a,method=name,group=label,n=int(mask.sum()),aircraft=len(set(keys[mask,0])),**dict(zip(['h1','h2','h5','h10'],e[mask,j].mean(0) if mask.any() else [np.nan]*4))))
    df=pd.DataFrame(rows);q=pd.DataFrame(pred)
    if t['args']['code']=='all_main':
        combined=[]
        df.to_csv(dest/'endpoints.csv',index=False)
        q.to_csv(dest/'angles_world.csv',index=False)
        ml=df[df.b.isin(['M','L'])]
        sweep_worlds=[*DEV,*CONF[:3]]
        for name,cells,panels,native in [
            ('tight_full',df[df.b=='T'],q,False),
            ('middle_loose_native',ml[ml.m==1.],q,True),
            ('middle_loose_sweep',ml[ml.w.isin(sweep_worlds)],q[q.w.isin(sweep_worlds)],False),
        ]:
            worlds=cells.w.nunique()
            assert len(cells)==14*worlds*cells.b.nunique()*cells.m.nunique()
            assert not cells.duplicated(['a','b','w','m']).any()
            folder=dest/name;folder.mkdir(exist_ok=True)
            summarize(folder,cells,panels,native_only=native)
            part=pd.read_csv(folder/'table1.csv')
            part['comparison']=name
            part['confirmation_worlds']=cells.loc[cells.w>=100,'w'].nunique()
            combined.append(part)
        pd.concat(combined,ignore_index=True).to_csv(dest/'table1.csv',index=False)
    else:
        summarize(dest,df,q)
    return dict(endpoints=len(df),world_panel_rows=len(q),folder=str(dest))

def summarize(dest,df,q,native_only=False):
    df.to_csv(dest/'endpoints.csv',index=False)
    q.to_csv(dest/'angles_world.csv',index=False)
    qc=q[q.w%3000>=100]
    qc.groupby(['method','group']).agg(**{k:(k,'mean') for k in ['h1','h2','h5','h10']},origins=('n','sum'),worlds=('n',lambda s:(s>0).sum()),aircraft=('aircraft','sum')).to_csv(dest/'angles.csv')
    fields=['risk','mbps','narrow','sense','unavailable'];out=[]
    for budget,g in df.groupby('b'):
        dev=g[g.w%3000<100];test=g[g.w%3000>=100]
        # Add an unchanged Chronos reference to subset experiments (including R09).
        ref='Cnative'
        if ref not in set(g.a):
            main=BASE/'results/main_T/endpoints.csv'
            if not main.exists():raise RuntimeError('main reference endpoints not yet available')
            source=pd.read_csv(main);source=source[source.a==ref]
            dev=pd.concat([dev,source[source.w<100]]);test=pd.concat([test,source[source.w>=100]])
        ref_risk=test[(test.a==ref)&(test.m==1)].risk.mean()
        policies=[('native',test)] if native_only else [
            ('native',test),('confirmation_common',test),('both_tuned',test),('dev_5pct',dev)]
        for desc,fit in policies:
            target,opts=workpoint(fit,ref,desc=='dev_5pct')
            if desc=='native':opts={a:(1.,1.,0.) for a in fit.a.unique()};target=None
            if desc=='confirmation_common' and target is not None and abs(target-ref_risk)<1e-9:opts[ref]=(1.,1.,0.)
            for a,block in test.groupby('a'):
                if a not in opts:
                    out.append(dict(b=budget,policy=desc,a=a,target=target,status='unreachable'));continue
                x,y,f=opts[a]
                for w,v in block.groupby('w'):
                    z=v.set_index('m');r=(1-f)*z.loc[x,fields]+f*z.loc[y,fields]
                    out.append(dict(b=budget,policy=desc,a=a,w=w,target=target,status='ok',left=x,right=y,prob_right=f,**r.to_dict()))
    rr=pd.DataFrame(out);rr.to_csv(dest/'policy_worlds.csv',index=False)
    rr.groupby(['b','policy','a'])[fields].mean().to_csv(dest/'communication.csv')
    from inference_stats import extend,adaptation
    extend(dest,df,q,rr);adaptation(BASE/'results')
