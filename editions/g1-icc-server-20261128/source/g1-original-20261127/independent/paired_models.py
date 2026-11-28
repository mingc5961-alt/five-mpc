"""Fit route, motion-only and fixed-fusion widths on disjoint local-world splits."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,multiprocessing,os,pickle,sys,time
from functools import lru_cache
import numpy as np
from history import ROOT,ASSETS,RUNTIME,PACKAGE,check_models,configure,put
from paired_grid import BASE,load_checked
from physical_inputs import SPLITS,sha_file
from slots import compute_slot

OUT=BASE/'aligned_models'
KINDS=('Acal','M','Fstar')
MOTION_COLUMNS=[0,1,3,4,8,9,10,11,12]
H=np.array([1.,2.,5.,10.])
INDEX=(H*2).astype(int)

def atomic_npz(path,**data):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**data)
    tmp.replace(path)

def trajectory(w,u,t):
    return np.stack([np.interp(t,w.times,w.positions[u,:,j]) for j in range(3)],axis=-1)

@lru_cache
def fixed_weights():
    return json.loads((OUT/'static.json').read_bytes())['weights']

def parts(pack,kind):
    import paid_full as pf
    if kind=='Acal':return pf.forecast(pack,'G',1)
    short,route,_,features,obs=pack
    weight=np.zeros(21) if kind=='M' else np.interp(pf.GRID,H,fixed_weights())
    center=short+weight[:,None]*(route-short)
    f=features.copy()
    f[:,3]=np.abs(pf.difference(pf.angle(center),pf.angle(obs))).ravel()
    f[:,5]=np.repeat(weight,2)
    return center,f[:,MOTION_COLUMNS] if kind=='M' else f

def source(job):
    import paid_full as pf
    from grudata import schedule
    split,k=job
    out=OUT/'inputs'/f'{k}.npz'
    if out.is_file():return
    with compute_slot():
        w,meta=load_checked(split,k)
        net=pf.model();states=[pf.State(w.cfg,net,'G') for _ in range(16)]
        sched=list(schedule(w,41000+k))
        eligible=[i for i,(_,t) in enumerate(sched) if t+10<=180]
        anchor=int(np.random.default_rng(271127+k).choice(eligible)) if split=='tube_calibration' else None
        sums=np.zeros((21,4));count=0;packets=[]
        for idx,(u,t) in enumerate(sched):
            st=states[u];net.n=st.count+1
            pack=st.update(w.observe(u,t),net)['G']
            if t+10>180:break
            if split in ('train','train_validation'):
                y=trajectory(w,u,t+H);s,r=pack[0][INDEX],pack[1][INDEX]
                pred=s[None]+np.linspace(0,1,21)[:,None,None]*(r-s)[None]
                sums+=np.arctan2(np.linalg.norm(np.cross(pred,y),axis=-1),np.sum(pred*y,axis=-1));count+=1
            elif anchor is None or idx==anchor:
                offsets=pf.GRID if anchor is None else np.arange(501)*.02
                packets.append((*pack,trajectory(w,u,t+offsets),t,u))
                if anchor is not None:break
        if split in ('train','train_validation'):
            atomic_npz(out,mean_loss=sums/count,origins=count,world=meta['physical_sha256'])
        else:
            data={key:np.stack(col) for key,col in zip(('s','r','g','f','obs','truth','t','u'),zip(*packets),strict=True)}
            atomic_npz(out,**data,world=meta['physical_sha256'])
        put(OUT/'origins'/f'{k}.json',dict(split=split,world=k,file_sha256=sha_file(out),physical_sha256=meta['physical_sha256']))

def entries(k):
    with np.load(OUT/'inputs'/f'{k}.npz') as d:
        columns=[d[name] for name in ('s','r','g','f','obs','truth')]
    for *packet,truth in zip(*columns,strict=True):
        yield tuple(packet),truth

def fit(kind):
    import paid_full as pf
    from lightgbm import LGBMRegressor
    from tube import raw,scores,EPS
    with compute_slot():
        x=[];ys=[]
        for k in SPLITS['tube_fit']:
            for pack,y in entries(k):
                c,feats=parts(pack,kind);x.append(feats)
                ys.append(np.log(np.maximum(EPS,np.abs(pf.difference(pf.angle(c),pf.angle(y))))).ravel())
        tree=LGBMRegressor(objective='quantile',alpha=.90,n_estimators=110,num_leaves=12,min_child_samples=85,
               learning_rate=.055,random_state=7,verbosity=-1,n_jobs=1)
        tree.fit(np.concatenate(x),np.concatenate(ys))
        out=tree.booster_;ratios=[]
        for k in SPLITS['tube_calibration']:
            [(pack,y)]=list(entries(k))
            c,f=parts(pack,kind);d=np.arange(501)*.02
            interp=np.stack([np.interp(d,pf.GRID,a) for a in c.T],axis=-1)
            err=np.abs(pf.difference(pf.angle(interp),pf.angle(y)))
            ratios.append(scores(err[None],raw(out,f[None]))[0])
        rank=int(np.ceil(.9*(len(ratios)+1)));factors=np.maximum.accumulate(np.sort(ratios,axis=0)[rank-1])
        assert np.isfinite(factors).all() and (factors>0).all()
        file=OUT/f'{kind}.pkl';tmp=file.with_suffix('.tmp')
        with tmp.open('wb') as stream:pickle.dump(dict(tree=out,final=factors,kind=kind,rank=rank,scores=ratios),stream)
        tmp.replace(file)
        return dict(kind=kind,width_model_sha256=sha_file(file),features=x[0].shape[1],train_origins=len(x),calibration_rank=rank,factors=factors.tolist())
@lru_cache
def tube(kind):
    with (OUT/f'{kind}.pkl').open('rb') as stream:return pickle.load(stream)
class Predictor:
    def __init__(self,w,kind):
        import paid_full as pf
        self.kind=kind;self.net=pf.model();self.states=[pf.State(w.cfg,self.net,'G') for _ in range(w.n)]
        self.bundle=tube(kind);self.factors=self.bundle['final']
    def predict(self,u,obs):
        import paid_full as pf
        st=self.states[u];self.net.n=st.count+1
        c,f=parts(st.update(obs,self.net)['G'],self.kind)
        return c,pf.raw(self.bundle['tree'],f)

def main():
    configure(ASSETS,RUNTIME);check_models(PACKAGE/'reference',ASSETS)
    import grurun,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for sub in ('inputs','origins'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    jobs=[(s,k) for s in ('train','train_validation','tube_fit','tube_calibration') for k in SPLITS[s]]
    put(OUT/'protocol.json',dict(fixed_weight_grid=[round(x*.05,2) for x in range(21)],horizons=H.tolist(),
        loss='world-equal mean geodesic LOS angle from original motion and route experts; fixed weights fit on train only',
        motion_columns=MOTION_COLUMNS,template='archived tube.py LightGBM fit, raw and scores; 1 prespecified random origin per calibration world',
        frozen_g1_source=sha_file(RUNTIME/'grurun.py'),train=list(SPLITS['train']),validation=list(SPLITS['train_validation']),
        fit_worlds=list(SPLITS['tube_fit']),calibration_worlds=list(SPLITS['tube_calibration'])))
    start=time.time();ctx=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(24,mp_context=ctx) as pool:
        for j in as_completed([pool.submit(source,j) for j in jobs]):j.result()
    def mean_loss(split):
        losses=[]
        for k in SPLITS[split]:
            with np.load(OUT/'inputs'/f'{k}.npz') as d:losses.append(d['mean_loss'])
        return np.mean(losses,axis=0)
    train=mean_loss('train');val=mean_loss('train_validation');indices=train.argmin(axis=0)
    put(OUT/'static.json',dict(weights=(indices/20).tolist(),validation_loss_rad=val[indices,np.arange(4)].tolist()))
    with ProcessPoolExecutor(3,mp_context=ctx) as pool:models=list(pool.map(fit,KINDS))
    put(OUT/'summary.json',dict(models=models,inputs=len(jobs),wall_s=time.time()-start,
                              static_weights=fixed_weights(),protocol_sha256=sha_file(OUT/'protocol.json')))
if __name__=='__main__':main()
