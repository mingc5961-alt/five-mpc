import hashlib
import pickle
import sys
from pathlib import Path
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.decomposition import PCA
from protocol import OUT,KINDS,log,write
from features import EXPAND,K
from refine import tubes
from routeclock.validity import EPS,QUERY_H,finite_quantile

def projection(z):
    # The h=0 row has no decoder state.
    rows=z[:,1:,-256:].reshape(-1,256)
    return PCA(n_components=8,svd_solver='full').fit(rows)

def inputs(common,special,pca):
    x=common
    if special is not None:
        if pca is not None:
            flat=special[..., -256:].reshape(-1,256)
            comps=pca.transform(flat).reshape(*special.shape[:-1],8)
            comps[:,0]=0.
            special=np.concatenate([special[...,:-256],comps],axis=-1)
        x=np.concatenate([x,special],axis=-1)
    n,h,_=x.shape
    axes=np.broadcast_to(np.eye(2),(n,h,2,2))
    return np.concatenate([np.repeat(x[:,:,None],2,axis=2),axes],axis=-1).reshape(n*h*2,-1)

def predict(head,x,z):
    design=inputs(x,z,head['pca'])
    pred=head['booster'].predict(design,num_threads=1)
    raw=np.maximum(np.exp(np.clip(pred,-30,5))-EPS,EPS).reshape(-1,len(K),2)
    return np.maximum.accumulate(raw,axis=1)[:,EXPAND]

def load_splits(model):
    data={}
    for split in ('train','calibration','validation'):
        with np.load(OUT/'features'/model/(split+'.npz')) as a:
            data[split]={k:a[k] for k in a.files}
    return data

def fit(model):
    folder=OUT/'heads'/model;folder.mkdir(parents=True,exist_ok=True)
    data=load_splits(model); train=data['train']
    hold=train['ids']%25==0
    y=np.log(EPS+np.maximum.accumulate(train['error'],axis=1)[:,np.rint(QUERY_H/.02).astype(int)])
    for kind in KINDS:
        special=kind=='special'
        def design(mask,pca):
            return inputs(train['common'][mask],train['special'][mask] if special else None,pca)
        def make_pca(mask):
            return projection(train['special'][mask]) if special and model=='VECTOR' else None
        pca=make_pca(~hold)
        a,b=design(~hold,pca),design(hold,pca)
        grid=[]
        for leaves,child in [(15,20),(15,50),(31,20),(31,50)]:
            reg=LGBMRegressor(num_leaves=leaves,min_child_samples=child,n_estimators=500,
                              n_jobs=1,random_state=7,verbosity=-1)
            reg.fit(a,y[~hold].ravel(),eval_set=[(b,y[hold].ravel())],callbacks=[early_stopping(30,verbose=False)])
            mse=np.mean((reg.booster_.predict(b,num_threads=1)-y[hold].ravel())**2)
            grid.append((float(mse),leaves,child,reg.best_iteration_))
        _,leaves,child,nt=min(grid)
        all_rows=np.ones(len(hold),dtype=bool);pca=make_pca(all_rows)
        reg=LGBMRegressor(num_leaves=leaves,min_child_samples=child,n_estimators=nt,n_jobs=1,random_state=7,verbosity=-1)
        reg.fit(design(all_rows,pca),y.ravel())
        head={'booster':reg.booster_,'pca':pca,'params':[leaves,child,nt],'kind':kind,'model':model}
        stats={'grid':grid,'chosen':[leaves,child,nt],'n_features':reg.n_features_in_}
        for split in ('calibration','validation'):
            d=data[split]; raw=predict(head,d['common'],d['special'] if special else None)
            scores=tubes.scores(d['error'],raw)
            if split=='calibration':
                assert len(scores)==400
                head['factors']=np.maximum.accumulate([finite_quantile(s,.1) for s in scores.T])
            cover=(scores<=head['factors'])
            width=np.rad2deg(raw[:,K[1:]]*head['factors'][None,:,None])
            stats[split]={'n_routes':len(set(d['ids'])),'n_anchors':len(raw),'coverage':cover.mean(0).tolist(),
                          'angular_halfwidth_deg':width.mean(0).tolist(),'factors':head['factors'].tolist()}
        tmp=folder/(kind+'.tmp');tmp.write_bytes(pickle.dumps(head));tmp.replace(folder/(kind+'.pkl'))
        write(folder/(kind+'.json'),stats)
        log(f"{model} {kind} 冻结完成；独立验证 5/10 秒双角前缀覆盖 {stats['validation']['coverage'][-3]:.3f}/{stats['validation']['coverage'][-1]:.3f}，树参数 {head['params']}。")
if __name__=='__main__': fit(sys.argv[1])
