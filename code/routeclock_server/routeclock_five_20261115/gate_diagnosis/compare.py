"""Matched CPU ablation. Validation selects hyperparameters by the original physical loss."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.ensemble import RandomForestClassifier
from extract import BASE,EXTRA,OUT,ROOT
from routeclock.geometry import angles,angle_error

BUDGET=(.0001,.001,.01)
FEATURES={'base':BASE,'plus':BASE+EXTRA,'horizon':['h_norm']}
def arrays(d):
    return tuple(d[[k+x for x in 'xyz']].to_numpy() for k in 'piy')
def errors(d,w):
    p,i,y=arrays(d);pred=i+w[:,None]*(p-i)
    angular=np.sum(np.rad2deg(angle_error(angles(pred),angles(y)))**2,axis=1)
    return angular,np.linalg.norm(pred-y,axis=1)
class Fit:
    def __init__(self,d,cols,kind,penalty):
        self.cols=cols;x=d[cols].to_numpy();self.mean=x.mean(0);self.std=np.maximum(x.std(0),1e-5)
        z=np.column_stack((np.ones(len(x)),(x-self.mean)/self.std))
        p,i,y=arrays(d);delta=p-i;truth=angles(y)
        npmean=lambda x:np.mean(x,axis=0)
        target=(d.imm_win==0).to_numpy(dtype=float)
        energy=(d.p_deg**2+d.i_deg**2).to_numpy()
        sw=1/(energy+np.median(energy)) if kind=='relative' else np.ones(len(d))
        scale=1. if kind=='original' else np.mean(sw*np.sum(angle_error(angles((p+i)/2),truth)**2,axis=1))
        def objective(b):
            t=z@b;w=expit(t)
            if kind=='winner':
                loss=np.mean(np.logaddexp(0,t)-target*t)/np.log(2);g=(w-target)/np.log(2)
            else:
                pred=i+w[:,None]*delta;xy2=np.sum(pred[:,:2]**2,axis=1);r=np.sqrt(xy2);rr=xy2+pred[:,2]**2
                e=angle_error(angles(pred),truth)
                da=(pred[:,0]*delta[:,1]-pred[:,1]*delta[:,0])/xy2
                de=(r*delta[:,2]-pred[:,2]*np.sum(pred[:,:2]*delta[:,:2],axis=1)/r)/rr
                loss=np.mean(sw*np.sum(e**2,axis=1))/scale
                g=2*sw*(e[:,0]*da+e[:,1]*de)*w*(1-w)/scale
            grad=z.T@g/len(d);grad[1:]+=2*penalty*b[1:]
            return loss+penalty*np.sum(b[1:]**2),grad
        start=np.random.default_rng(7).normal(0,.01,z.shape[1])
        opt=minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
        if not opt.success:raise RuntimeError(opt.message)
        self.b=opt.x
    def weights(self,d):
        return expit(self.b[0]+((d[self.cols].to_numpy()-self.mean)/self.std)@self.b[1:])
def fit(tr,val,fs,obj):
    if obj=='winner_forest':
        # A shallow nonlinear probe tests whether a linear gate is hiding usable evidence.
        forest=RandomForestClassifier(n_estimators=120,max_depth=3,min_samples_leaf=6,max_features=1.,random_state=7,n_jobs=1)
        forest.fit(tr[FEATURES[fs]],tr.imm_win)
        class Probe:
            def weights(self,d):return 1-forest.predict_proba(d[FEATURES[fs]])[:,1]
        return Probe(),None
    candidates=[Fit(tr,FEATURES[fs],obj,reg) for reg in BUDGET]
    loss=[errors(val,m.weights(val))[0].mean() for m in candidates];j=int(np.argmin(loss))
    return candidates[j],BUDGET[j]

def score(d,w,scores=None):
    ae,me=errors(d,w);im=d.imm_win.to_numpy(dtype=bool)
    s=1-w if scores is None else scores
    call=w<.5;tp=(im&call).sum()
    return dict(n=len(d),positives=im.sum(),called=call.sum(),tp=tp,
         precision=tp/call.sum() if call.sum() else np.nan,recall=tp/im.sum() if im.sum() else np.nan,
         auc=roc_auc_score(im,s) if np.unique(im).size==2 else np.nan,ap=average_precision_score(im,s),
         mean_weight=np.mean(w),mse=ae.mean(),rms=np.sqrt(ae.mean()),metres=me.mean(),
         hard_mse=np.mean(np.where(call,d.i_deg**2,d.p_deg**2)),rank_prevalence=im.mean(),
         within_horizon_auc=np.mean([roc_auc_score(im[d.h.to_numpy()==h],s[d.h.to_numpy()==h])
                                     for h in d.h.unique() if len(np.unique(im[d.h.to_numpy()==h]))==2]))
def rows_for(d,size):
    return pd.concat([d[d.split=='train'],d[(d.split=='extra_train')&(d['index']<size-12)]]).sort_values(['split','index','step','h'])
def main():
    files=sorted((OUT/'routes').glob('*.csv'));assert len(files)==80
    allrows=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in files],ignore_index=True)
    assert allrows.groupby('route').route_hash.first().nunique()==84
    d=allrows[allrows.step==179];old=d[d.split=='train'];val=d[d.split=='validation']
    assert len(old)==84 and old.imm_win.sum()==25
    screens={'network':d[d.split=='network'],'audit':d[d.split=='audit']}
    scores=[];predictions=[]
    designs=[('base',k) for k in ('original','normalized','relative','winner','winner_forest')]+[('plus',k) for k in ('original','normalized','relative','winner','winner_forest')]+[('horizon','winner')]
    for size,windows in [(12,1),(24,1),(48,1),(12,3)]:
        tr=rows_for(allrows if windows==3 else d,size)
        const_grid=np.linspace(0,1,101);fixed=min(const_grid,key=lambda v:errors(tr,np.full(len(tr),v))[0].mean())
        for fs,obj in designs+[('none','route'),('none','imm'),('none','constant'),('oracle','winner_oracle')]:
            model=reg=None
            if fs in FEATURES:model,reg=fit(tr,val,fs,obj)
            for split,test in screens.items():
                w=(model.weights(test) if model else (1-test.imm_win.to_numpy() if fs=='oracle' else
                    np.full(len(test),{'route':1.,'imm':0.,'constant':fixed}.get(obj,1.))))
                identity=dict(size=size,windows=windows,features=fs,objective=obj,split=split)
                scores.append(dict(**identity,penalty=reg,**score(test,w)))
                ae,_=errors(test,w)
                for route,step,h,ww,ee in zip(test.route,test.step,test.h,w,ae):
                    predictions.append(dict(**identity,route=route,step=step,h=h,route_weight=ww,deg2=ee))
    pd.DataFrame(scores).to_csv(OUT/'comparisons.csv',index=False)
    pred=pd.DataFrame(predictions);pred.to_csv(OUT/'predictions.csv',index=False)
    # Out-of-route predictions for *all* 84 original queries, not a classifier fitted to these 25.
    oof=old.copy()
    for fs in ('base','plus','horizon'):
        oof['p_imm_'+fs]=np.nan
        for route,g in old.groupby('route'):
            model,_=fit(old[old.route!=route],val,fs,'winner')
            oof.loc[g.index,'p_imm_'+fs]=1-model.weights(g)
    oof.to_csv(OUT/'original84.csv',index=False,encoding='utf-8-sig')
    oof[oof.imm_win==1].sort_values('gain_deg2',ascending=False).to_csv(OUT/'imm_better25.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(features=fs,**score(oof,1-oof['p_imm_'+fs].to_numpy())) for fs in ('base','plus','horizon')]).to_csv(OUT/'leave_route_out.csv',index=False)
    # Single observable ranking; sign selected only on original training routes.
    signals=[]
    for name in BASE+EXTRA:
        a=roc_auc_score(old.imm_win,old[name]);sign=1 if a>=.5 else -1
        for split,test in [('train',old),*screens.items()]:
            signals.append(dict(feature=name,split=split,sign=sign,auc=roc_auc_score(test.imm_win,sign*test[name]),
                   win_median=test.loc[test.imm_win==1,name].median(),other_median=test.loc[test.imm_win==0,name].median()))
    pd.DataFrame(signals).to_csv(OUT/'single_signals.csv',index=False)
    rules=[]
    for split,g in d.groupby('split'):
        for name,switch in [('low_support',g.support<.5),('replay',g.replay_gap_deg>0),
                   ('agreement',(g.support<.5)&(g.replay_gap_deg>0)&(g.replay_imm_fraction>.5))]:
            rules.append(dict(split=split,rule=name,**score(g,(~switch).to_numpy(float))))
    pd.DataFrame(rules).to_csv(OUT/'causal_rules.csv',index=False)
    named=oof[oof.imm_win==1].sort_values('gain_deg2',ascending=False).copy()
    named['原门控路线权重']=pickle.loads((ROOT/'runs/five_pilot/results/bundle.pkl').read_bytes())['gates']['RC'].weights(named[BASE].to_numpy())
    columns={'route':'航线','group':'仿真行为（仅分析）','t':'观测时刻_秒','h':'预测时长_秒','p_deg':'事后路线角误差_度',
        'i_deg':'事后IMM角误差_度','gain_deg2':'选IMM可省_平方度','innovation':'预测前归一化路径匹配残差',
        'support':'预测前路线支持度','replay_gap_deg':'已兑现六次平均误差差_度_正值偏IMM',
        'replay_imm_fraction':'已兑现六次IMM胜率','p_imm_base':'留整条航线_原特征IMM分数',
        'p_imm_plus':'留整条航线_扩展特征IMM分数','原门控路线权重':'原门控路线权重'}
    named[list(columns)].rename(columns=columns).to_csv(OUT/'25次IMM更准_逐样本.csv',index=False,encoding='utf-8-sig')
    # Stratified alternative training subsets from the same 48 independent routes, fixed audit data.
    pool=rows_for(d,48);stable=[]
    for seed in [7,11,19,23,31]:
        rng=np.random.default_rng(seed)
        orders=[rng.permutation(g.route.unique()) for _,g in pool.groupby('group')]
        for size in (12,24,48):
            ids=np.concatenate([o[:size//6] for o in orders]);tr=pool[pool.route.isin(ids)]
            for fs,obj in [('base','original'),('plus','winner')]:
                model,_=fit(tr,val,fs,obj);test=screens['audit']
                stable.append(dict(seed=seed,size=size,features=fs,objective=obj,**score(test,model.weights(test))))
    pd.DataFrame(stable).to_csv(OUT/'resampled_training.csv',index=False)
    # Task-paired uncertainty on independently generated single-vehicle audit tasks.
    a=pred[(pred.split=='audit')&(pred.windows==1)].groupby(['size','features','objective','route']).deg2.mean().unstack('route')
    ci=[]
    for key in a.index:
        for ref in [(12,'base','original'),(48,'none','route')]:
            delta=(a.loc[key]-a.loc[ref]).to_numpy();rng=np.random.default_rng(7)
            q=np.quantile(delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1),[.025,.975])
            ci.append(dict(size=key[0],features=key[1],objective=key[2],reference=str(ref),delta=delta.mean(),low=q[0],high=q[1]))
    pd.DataFrame(ci).to_csv(OUT/'paired_audit.csv',index=False)
    with (ROOT/'runs/five_pilot/results/bundle.pkl').open('rb') as f:gate=pickle.load(f)['gates']['RC']
    fitted,_=fit(old,val,'base','original')
    np.testing.assert_allclose(gate.weights(old[BASE].to_numpy()),fitted.weights(old),rtol=0,atol=2e-6)
    print('Verified 25 wins, old gate, nested learning curves, grouped validation and independent audit.')
if __name__=='__main__':main()

