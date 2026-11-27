import itertools,numpy as np,pandas as pd
from base import BASE
def boot(test,budget):
    labels=sorted(test.a.unique());worlds=sorted(test.w.unique());scales=sorted(test.m.unique())
    cols=['risk','mbps','narrow'];n=len(worlds);ref=labels.index('Cnative')
    z=test.set_index(['w','a','m'])[cols].reindex(pd.MultiIndex.from_product([worlds,labels,scales])).to_numpy().reshape(n,len(labels),len(scales),3)
    assert np.isfinite(z).all()
    rng=np.random.default_rng(270125);wt=rng.multinomial(n,np.full(n,1/n),size=2000)/n
    z=np.einsum('kw,wasc->kasc',wt,z)
    risk=z[:,:,:,0];native=risk[:,ref,scales.index(1.)]
    lo=risk.min(2).max(1);hi=risk.max(2).min(1);valid=lo<=hi
    target=np.clip(native,lo,hi);out=[]
    for policy in ('confirmation_common','both_tuned'):
        allz=[]
        for j in range(len(labels)):
            best=np.full(len(wt),-np.inf);chosen=np.full((len(wt),3),np.nan)
            for i,k in itertools.combinations_with_replacement(range(len(scales)),2):
                x,y=risk[:,j,i],risk[:,j,k];den=y-x
                f=np.divide(target-x,den,out=np.zeros_like(x),where=abs(den)>1e-12)
                ok=valid&(target>=np.minimum(x,y)-1e-9)&(target<=np.maximum(x,y)+1e-9)
                vals=z[:,j,i]+np.clip(f,0,1)[:,None]*(z[:,j,k]-z[:,j,i])
                ix=ok&(vals[:,1]>best);chosen[ix]=vals[ix];best[ix]=vals[ix,1]
            if j==ref and policy=='confirmation_common':
                exact=valid&(abs(native-target)<1e-9);chosen[exact]=z[exact,j,scales.index(1.)]
            allz.append(chosen)
        for j,a in enumerate(labels):
            for c,name in enumerate(cols):
                diff=allz[j][:,c]-allz[ref][:,c];ok=np.isfinite(diff)
                ci=np.quantile(diff[ok],[.025,.975]) if ok.any() else [np.nan,np.nan]
                out.append(dict(b=budget,policy=policy,a=a,metric=name,lo95=ci[0],hi95=ci[1],infeasible_fraction=1-ok.mean(),worlds=n))
    return out
def extend(dest,df,q,rr):
    ci=[]
    for budget,g in df.groupby('b'):
        test=g[g.w%3000>=100]
        if 'Cnative' not in set(test.a):
            base=pd.read_csv(BASE/'results/main_T/endpoints.csv')
            test=pd.concat([test,base[(base.a=='Cnative')&(base.w>=100)]])
        if (rr.policy=='both_tuned').any():
            ci.extend(boot(test,budget))
    for (b,policy),g in rr[rr.policy.isin(['native','dev_5pct'])&rr.status.eq('ok')].groupby(['b','policy']):
        for field in ['risk','mbps','narrow']:
            p=g.pivot(index='w',columns='a',values=field)
            if 'Cnative' not in p:continue
            for a in p:
                d=(p[a]-p.Cnative).dropna().to_numpy()
                if not len(d):continue
                ix=np.random.default_rng(270125).integers(len(d),size=(2000,len(d)))
                l,u=np.quantile(d[ix].mean(1),[.025,.975])
                ci.append(dict(b=b,policy=policy,a=a,metric=field,lo95=l,hi95=u,worlds=len(d),infeasible_fraction=0))
    pd.DataFrame(ci).to_csv(dest/'paired_ci.csv',index=False)
    angles=q[(q.w%3000>=100)&(q.group=='all')&(q.a==q.method)].groupby('a')[['h1','h2','h5','h10']].mean().reset_index()
    comm=rr[rr.status=='ok'].groupby(['b','policy','a'])[['risk','mbps','narrow','sense','unavailable']].mean().reset_index()
    comm['geometric']=comm.risk-comm.unavailable
    comm.merge(angles,on='a',how='left').to_csv(dest/'table1.csv',index=False)
def adaptation(root):
    # Choose a candidate and scalar policy exclusively on development worlds.
    # The paired confirmation return is read only after the choice is made.
    dirs=['main_T','training_original','training_half']
    if not all((root/x/'table1.csv').exists() for x in dirs):return
    sources={k:pd.read_csv(root/k/'endpoints.csv') for k in dirs}
    devs=[];valid=[]
    from tables import workpoint
    for key,x in sources.items():
        dv=x[(x.w<100)&x.a.eq('G8P')]
        _,opts=workpoint(dv,'G8P',True)
        if 'G8P' not in opts:continue
        l,r,f=opts['G8P'];means=dv.groupby('m').mbps.mean()
        devs.append(((1-f)*means[l]+f*means[r],key,l,r,f))
    if not devs:
        (root/'adaptation_no_feasible.txt').write_text('开发世界的5%风险约束下无可行GRU门控端点。');return
    g,key,l,r,f=max(devs)
    test=sources[key].query("w>=100 and a=='G8P'")
    for w,block in test.groupby('w'):
        v=block.set_index('m')
        valid.append(dict(w=w,training=key,dev_mbps=g,left=l,right=r,prob_right=f,**{c:(1-f)*v.loc[l,c]+f*v.loc[r,c] for c in ['risk','mbps','narrow','closed1','closed2','closed5','closed10']}))
    pd.DataFrame(valid).to_csv(root/'development_selected_GRU_confirmation.csv',index=False)
