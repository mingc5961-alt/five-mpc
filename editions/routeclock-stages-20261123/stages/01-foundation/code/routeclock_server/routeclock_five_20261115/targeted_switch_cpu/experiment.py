from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from prepare import OUT,HERE,extract
from compare import errors,fit,BASE
FEATURES=['h_norm','support','innovation','replay_gap_deg','replay_imm_fraction','uncertainty_ratio','disagreement']
SIGNS={'innovation':1,'replay_gap_deg':1,'replay_imm_fraction':1,'uncertainty_ratio':1,'disagreement':-1}
PRIMARY='switch_target'
def load(paths):
    df=pd.concat([pd.read_csv(p) for p in sorted(paths)],ignore_index=True)
    df['qid']=df.route+'@'+df.step.astype(str)+'+'+df.h.astype(str)
    assert df.qid.is_unique
    df['fixed_err']=errors(df,np.full(len(df),.89))[0]
    df['gain_fixed']=df.fixed_err-df.i_deg**2
    return df

def match(data,limit=None):
    low=data[data.support<.5]
    a=low[low.gain_deg2>=1];b=low[low.gain_deg2<=-4]
    edges=a.merge(b,on=['step','h'],suffixes=('_a','_b'))
    edges['distance']=(edges.support_a-edges.support_b).abs()
    edges=edges[(edges.route_a!=edges.route_b)&(edges.distance<=.05)].copy()
    if edges.empty:return pd.DataFrame(columns=['pair','side',*data.columns])
    edges['key']=[tuple(sorted(pair)) for pair in zip(edges.route_a,edges.route_b)]
    edges=edges.sort_values(['distance','qid_a','qid_b']).drop_duplicates('key').reset_index(drop=True)
    routes=sorted(set(edges.route_a)|set(edges.route_b));idx={r:i for i,r in enumerate(routes)}
    n=len(edges);positions=list(range(n))*3
    indices=[idx[r] for r in edges.route_a]+[idx[r] for r in edges.route_b]+[len(routes)]*n
    matrix=coo_matrix((np.ones(len(indices)),(indices,positions)),shape=(len(routes)+1,n)).tocsr()
    upper=np.r_[np.ones(len(routes)),limit or len(routes)]
    # First maximize cardinality, then minimize support differences. Stable tie order.
    cost=-(n+1)+edges.distance.to_numpy()+1e-9*np.arange(n)/max(n,1)
    result=milp(cost,integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,0,upper),
                options={'mip_rel_gap':0.0})
    if not result.success:raise RuntimeError(result.message)
    lookup=data.set_index('qid',drop=False);out=[]
    for j,edge in enumerate(edges[result.x>.5].itertuples()):
        for side,key in [('benefit',edge.qid_a),('harm',edge.qid_b)]:
            out.append(dict(pair=j+1,side=side,**lookup.loc[key].to_dict()))
    matched=pd.DataFrame(out)
    assert matched.route.is_unique
    return matched

def score_pairs(tr,test):
    output=[];examples=[]
    for dataset,d in [('train_LOPO',tr),('fresh_test',test)]:
        if d.empty:continue
        d=d.copy()
        for pair,g in d.groupby('pair'):
            t=tr[tr.pair!=pair] if dataset=='train_LOPO' else tr
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.25,max_iter=1000))
            model.fit(t[FEATURES],t.side=='benefit')
            d.loc[g.index,'paired_score']=model.predict_proba(g[FEATURES])[:,1]
        examples.append(d.assign(dataset=dataset))
        for f,sign in dict(SIGNS,paired_score=1).items():
            v=d.pivot(index='pair',columns='side',values=f)
            diff=(v.benefit-v.harm)*sign
            output.append(dict(dataset=dataset,feature=f,pairs=len(v),correct=int((diff>0).sum()),
                         ties=int((diff==0).sum()),concordance=np.mean((diff>0)+.5*(diff==0)),
                         median_benefit=v.benefit.median(),median_harm=v.harm.median(),
                         auc=roc_auc_score(d.side=='benefit',sign*d[f])))
    joined=pd.concat(examples)
    joined.to_csv(OUT/'matched_cases.csv',index=False,encoding='utf-8-sig')
    named=joined[['dataset','pair','side','route','t','h','support','innovation','replay_gap_deg',
                  'replay_imm_fraction','uncertainty_ratio','disagreement','paired_score','p_deg','i_deg','gain_deg2']].copy()
    named['dataset']=named.dataset.map({'train_LOPO':'训练配对（留整对打分）','fresh_test':'独立测试配对'})
    named['side']=named.side.map({'benefit':'IMM明显获益','harm':'IMM严重受损'})
    named.columns=['来源','配对编号','实际结果','航线','观测截止秒','预测时长秒','当时路线支持度','匹配残差',
                   '过去路线误差减IMM误差_度','过去IMM胜率','位置协方差迹的对数比','专家角分歧_弧度',
                   '分类器获益分数','真实路线角误差_度','真实IMM角误差_度','切到IMM可挽回_平方度']
    named.to_csv(OUT/'低支持同条件配对样本.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(output).to_csv(OUT/'paired_signals.csv',index=False)

class Switch:
    def __init__(self,tr,val):
        self.model=RandomForestRegressor(n_estimators=200,max_depth=3,min_samples_leaf=6,n_jobs=1,random_state=7)
        low=tr[tr.support<.5]
        self.model.fit(low[FEATURES],low.gain_fixed)
        s=self.model.predict(val[FEATURES]);grid=[0.,1.,4.,np.inf]
        def utility(thr):
            call=(val.support<.5)&(s>thr)
            return (float(val.loc[call,'gain_fixed'].sum()),-int(call.sum()),thr)
        self.threshold=max(grid,key=utility)

def ledger(d,weights,model,split,threshold=np.nan,hard=False):
    e=errors(d,weights)[0];gain=d.fixed_err.to_numpy()-e
    p=(d.p_deg**2).to_numpy();i=(d.i_deg**2).to_numpy()
    switch=weights==0 if hard else np.zeros(len(d),dtype=bool)
    if hard:
        gf=gain[switch];gp=p[switch]-i[switch]
        assert np.allclose(gain[~switch],0,atol=1e-8)
        assert np.isclose(gf.sum(),gain.sum())
    else:gf=gp=np.array([])
    return dict(model=model,split=split,n=len(d),routes=d.route.nunique(),mse=e.mean(),route_weight=weights.mean(),
        saved=np.maximum(gain,0).sum(),lost=np.maximum(-gain,0).sum(),net=gain.sum(),mean_net=gain.mean(),
        calls=switch.sum() if hard else np.nan,tp_p=(gp>0).sum() if hard else np.nan,fp_p=(gp<0).sum() if hard else np.nan,
        saved_switch_p=gp.clip(0).sum() if hard else np.nan,lost_switch_p=(-gp).clip(0).sum() if hard else np.nan,
        tp_fixed=(gf>0).sum() if hard else np.nan,fp_fixed=(gf<0).sum() if hard else np.nan,
        saved_switch_fixed=gf.clip(0).sum() if hard else np.nan,lost_switch_fixed=(-gf).clip(0).sum() if hard else np.nan,
        threshold=threshold)

def main():
    old=load((extract.ROOT/'gate_diagnosis/results/routes').glob('*.csv'))
    fresh=load((OUT/'routes').glob('*.csv'))
    tr=old[old.split.isin(['train','extra_train'])];vold=old[(old.split=='validation')&(old.step==179)]
    pool=fresh[fresh.split=='target_pool'];val=fresh[fresh.split=='target_val'];test=fresh[fresh.split=='target_test']
    legacy=old[(old.split=='network')&(old.step==179)];audit=old[(old.split=='audit')&(old.step==179)]
    assert [g.route.nunique() for g in (tr,pool,val,test)]==[48,96,24,48]
    assert fresh.groupby('route').route_hash.first().is_unique
    assert not (set(fresh.route_hash)&set(old.route_hash))
    trainpairs=match(pool,12)
    assert len(trainpairs)>=4
    target_routes=trainpairs.route.tolist();count=len(target_routes)
    registry=pool[['route','group']].drop_duplicates().sort_values('route')
    arms={'base':[], 'target':target_routes}
    for seed in [7,11,19,23,31]:
        r=np.random.default_rng(seed);arms['random'+str(seed)]=r.choice(registry.route,count,replace=False).tolist()
    rng=np.random.default_rng(7)
    arms['type_matched']=[r for g,n in trainpairs.group.value_counts().items()
              for r in rng.choice(registry.loc[registry.group==g,'route'],n,replace=False)]
    records=[];preds=[];selection=[]
    screens={'validation':val,'natural':test,'natural_low':test[test.support<.5],'last_origin':test[test.step==179],
             'old6':legacy,'old24':audit}
    def evaluate(name,fun,threshold=np.nan,hard=False):
        for split,d in screens.items():
            w=fun(d);records.append(ledger(d,w,name,split,threshold,hard))
            if split=='natural':
                e=errors(d,w)[0]
                preds.append(pd.DataFrame(dict(qid=d.qid,route=d.route,model=name,weight=w,deg2=e,
                       gain_fixed=d.fixed_err-e,gain_route=d.p_deg**2-e,expert_gain=d.gain_deg2)))
    prev,_=fit(tr[tr.step==179],vold,'base','original')
    evaluate('current48',prev.weights)
    evaluate('fixed89',lambda d:np.full(len(d),.89))
    evaluate('route',lambda d:np.ones(len(d)))
    for name,routes in arms.items():
        aug=pd.concat([tr,pool[pool.route.isin(routes)]])
        for r in routes:selection.append(dict(arm=name,route=r))
        gate,_=fit(aug,val,'base','original')
        evaluate('soft_'+name,gate.weights)
        switch=Switch(aug,val)
        def weight(d,thr):
            call=(d.support.to_numpy()<.5)&(switch.model.predict(d[FEATURES])>thr)
            return np.where(call,0.,.89)
        evaluate('switch_'+name,lambda d:weight(d,switch.threshold),switch.threshold,True)
        if name in ['base','target']:
            evaluate('zero_'+name,lambda d:weight(d,0.),0.,True)
    pair_model=make_pipeline(StandardScaler(),LogisticRegression(C=.25,max_iter=1000))
    pair_model.fit(trainpairs[FEATURES],trainpairs.side=='benefit')
    def pair_weight(d,thr):
        call=(d.support.to_numpy()<.5)&(pair_model.predict_proba(d[FEATURES])[:,1]>thr)
        return np.where(call,0.,.89)
    def pair_utility(thr):
        w=pair_weight(val,thr)
        return (val.loc[w==0,'gain_fixed'].sum(),-int((w==0).sum()),thr)
    chosen=max((.5,.65,.8,1.),key=pair_utility)
    evaluate('pair_classifier',lambda d:pair_weight(d,chosen),chosen,True)
    evaluate('pair_classifier_half',lambda d:pair_weight(d,.5),.5,True)
    score_pairs(trainpairs,match(test))
    df=pd.DataFrame(records);pd.DataFrame(selection).to_csv(OUT/'training_routes.csv',index=False)
    df.to_csv(OUT/'ledger.csv',index=False)
    pred=pd.concat(preds);pred.to_csv(OUT/'query_predictions.csv',index=False)
    accounts=[]
    for ref in ['fixed89','current48','route']:
        default=pred[pred.model==ref].set_index('qid').deg2
        for name,g in pred.groupby('model'):
            base_errors=g.qid.map(default).to_numpy()
            actual=base_errors-g.deg2.to_numpy()
            switch=g.weight.to_numpy()==0
            for mask,label in [(np.ones(len(g),bool),'all'),(switch,'switched'),(~switch,'remaining')]:
                changes=actual[mask]
                accounts.append(dict(model=name,baseline=ref,subset=label,n=len(changes),
                    saved=np.maximum(changes,0).sum(),lost=np.maximum(-changes,0).sum(),
                    net=changes.sum(),improved=(changes>1e-9).sum(),worsened=(changes< -1e-9).sum()))
    pd.DataFrame(accounts).to_csv(OUT/'pairwise_accounts.csv',index=False)
    reference=pred.pivot(index=['route','qid'],columns='model',values='deg2').groupby(level='route').mean()
    boot=np.random.default_rng(7).integers(0,len(reference),(10000,len(reference)))
    ci=[]
    for method in reference.columns:
        for base in ['fixed89','current48','switch_base','switch_random7','soft_target']:
            delta=(reference[base]-reference[method]).to_numpy()
            lo,hi=np.quantile(delta[boot].mean(1),[.025,.975]);jack=(delta.sum()-delta)/(len(delta)-1)
            ci.append(dict(method=method,baseline=base,mean_gain=delta.mean(),lo=lo,hi=hi,
                      loo_min=jack.min(),loo_max=jack.max(),routes_better=(delta>1e-9).sum(),routes_worse=(delta< -1e-9).sum()))
    pd.DataFrame(ci).to_csv(OUT/'paired_intervals.csv',index=False)
    summary=[]
    for name,d in [('base',tr),('pool',pool),('target_add',pool[pool.route.isin(target_routes)]),('validation',val),('test',test)]:
        low=d[d.support<.5];summary.append(dict(split=name,routes=d.route.nunique(),queries=len(d),low=len(low),
                gain1=(low.gain_deg2>=1).sum(),harm4=(low.gain_deg2<=-4).sum(),
                gain_routes=low[low.gain_deg2>=1].route.nunique(),harm_routes=low[low.gain_deg2<=-4].route.nunique()))
    pd.DataFrame(summary).to_csv(OUT/'population.csv',index=False)
    print(df[(df.split=='natural')&df.model.isin(['fixed89','current48','switch_base','switch_target','zero_target','soft_target'])].to_string(index=False))
if __name__=='__main__':main()

