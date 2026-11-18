"""Sensitivity to horizon concentration in the primary support-matched pairs."""
from experiment import *
def diversified(d):
    a=d[(d.support<.5)&(d.gain_deg2>=1)]
    b=d[(d.support<.5)&(d.gain_deg2<=-4)]
    e=a.merge(b,on=['step','h'],suffixes=('_a','_b'))
    e['gap']=(e.support_a-e.support_b).abs()
    e=e[(e.route_a!=e.route_b)&(e.gap<=.05)].copy()
    e['key']=[tuple(sorted(p)) for p in zip(e.route_a,e.route_b)]
    e=e.sort_values(['gap','qid_a','qid_b']).drop_duplicates(['key','h']).reset_index(drop=True)
    hs=sorted(e.h.unique());routes=sorted(set(e.route_a)|set(e.route_b))
    n=len(e);extra=len(hs)*3;rows=[];low=[];high=[]
    def constrain(row,l,u):rows.append(row);low.append(l);high.append(u)
    for r in routes:constrain(np.r_[(e.route_a==r)|(e.route_b==r),np.zeros(extra)],0,1)
    constrain(np.r_[np.ones(n),np.zeros(extra)],12,12)
    rewards=[]
    for j,h in enumerate(hs):
        for k,reward in enumerate([100.,10.,1.],start=1):
            row=np.r_[(e.h==h).astype(float),np.zeros(extra)];row[n+3*j+k-1]=-k
            constrain(row,0,np.inf);rewards.append(reward)
    cost=np.r_[e.gap+np.arange(n)*1.e-9/max(n,1),-np.array(rewards)]
    result=milp(cost,integrality=np.ones(n+extra),bounds=Bounds(0,1),
                constraints=LinearConstraint(np.array(rows),low,high),options={'mip_rel_gap':0.})
    if not result.success:raise RuntimeError(result.message)
    matched=[];look=d.set_index('qid',drop=False)
    for j,r in enumerate(e[result.x[:n]>.5].itertuples()):
        for side,q in [('benefit',r.qid_a),('harm',r.qid_b)]:
            matched.append(dict(pair=j+1,side=side,**look.loc[q].to_dict()))
    return pd.DataFrame(matched)
def main():
    new=load((OUT/'routes').glob('*.csv'));pool=new[new.split=='target_pool']
    test=new[new.split=='target_test'];val=new[new.split=='target_val']
    old=load((extract.ROOT/'gate_diagnosis/results/routes').glob('*.csv'))
    base=old[old.split.isin(['train','extra_train'])]
    pairs=diversified(pool);assert pairs.route.is_unique
    pairs.to_csv(OUT/'sensitivity_pairs.csv',index=False)
    aug=pd.concat([base,pool[pool.route.isin(pairs.route)]])
    forest=Switch(aug,val);gate,_=fit(aug,val,'base','original')
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.25,max_iter=1000)).fit(pairs[FEATURES],pairs.side=='benefit')
    probability=lambda d:clf.predict_proba(d[FEATURES])[:,1]
    weights=lambda d,s,thr:np.where((d.support<.5)&(s>thr),0,.89)
    def selection(t):
        w=weights(val,probability(val),t);return (val.loc[w==0,'gain_fixed'].sum(),-int((w==0).sum()),t)
    prob_threshold=max([.5,.65,.8,1.],key=selection)
    r=[]
    for key,d in [('val',val),('natural',test)]:
        for m,thr,s in [('switch',forest.threshold,forest.model.predict(d[FEATURES])),('zero',0.,forest.model.predict(d[FEATURES])),
                        ('classifier',prob_threshold,probability(d)),('classifier_half',.5,probability(d))]:
            r.append(ledger(d,weights(d,s,thr),m,key,thr,True))
        r.append(ledger(d,gate.weights(d),'soft',key))
    pd.DataFrame(r).to_csv(OUT/'sensitivity_ledger.csv',index=False)
    oldgate,_=fit(base[base.step==179],old[(old.split=='validation')&(old.step==179)],'base','original')
    e=errors(test,gate.weights(test))[0]
    checks=[]
    avg=float(gate.weights(val).mean())
    opt=min(np.linspace(0,1,101),key=lambda w:errors(aug,np.full(len(aug),w))[0].mean())
    for name,v in [('fixed89',test.fixed_err),('current48',errors(test,oldgate.weights(test))[0]),
                   (f'validation_mean_weight_{avg:.5f}',errors(test,np.full(len(test),avg))[0]),
                   (f'training_best_constant_{opt:.2f}',errors(test,np.full(len(test),opt))[0])]:
        delta=pd.Series(np.asarray(v)-e,index=test.route).groupby(level=0).mean().to_numpy()
        draws=np.random.default_rng(7).integers(0,len(delta),(10000,len(delta)))
        lo,hi=np.quantile(delta[draws].mean(axis=1),[.025,.975])
        checks.append(dict(baseline=name,mean_gain=delta.mean(),lo=lo,hi=hi))
    pd.DataFrame(checks).to_csv(OUT/'sensitivity_soft_intervals.csv',index=False)
    print(pairs[pairs.side=='benefit'].h.value_counts().to_dict())
    print(pd.DataFrame(r).query("split=='natural'")[['model','mse','threshold','calls','tp_fixed','net']])
if __name__=='__main__':main()

