"""Endpoint-only RC center ablations. Fixed recipe, frozen original gate, no feedback tuning."""
from pathlib import Path
import gzip
import pickle
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'专用尺度管实验'))
from protocol import ROOT
from features import apply_gate
from advanced_tubes import Predictor
from routeclock.geometry import angles,angle_error
from routeclock.config import load_config
from refine import network,world_cache
from refine.data import ARCHIVE,BASE
CFG=load_config();CFG['simulation']['sensing_budget']=.05/3
CACHE=ROOT/'work/baseline_cache'
GATE=pickle.loads((CACHE/'classical.pkl').read_bytes())['RC']['gate']
HS=np.array([1,3,5,10])
IDX=HS*2
ALPHA=np.arange(101)/100

def rows(package,features):
    apply_gate(package,GATE,features,'RC','P')
    imm,route,rc=[package.centers[k] for k in ('I','P','RC')]
    x=np.column_stack([features[IDX],HS,package.diagnostics['gate_weights']['RC'][IDX],
               np.full(4,package.diagnostics['recent_spacing_s']),angle_error(angles(route[IDX]),angles(imm[IDX])),
               angle_error(angles(imm[IDX]),angles(imm[:1])),angle_error(angles(route[IDX]),angles(route[:1])),
               np.linalg.norm(route[IDX],axis=1)-np.linalg.norm(imm[IDX],axis=1),
               np.linalg.norm(route[IDX],axis=1)-np.linalg.norm(route[:1])])
    return x,imm[IDX],route[IDX],rc[IDX]

def corpus(split):
    blocks=[[] for _ in range(5)];labels=[]
    for file in sorted((ARCHIVE/split).glob('*.pkl')):
        route=int(file.stem)
        if split=='train' and route%5==0:continue
        with np.load(BASE/'observations'/split/(file.stem+'.npz')) as d: times=d['times']
        for j,r in enumerate(pickle.loads(file.read_bytes())):
            p=r['package']
            history=times[:times.searchsorted(p.origin,side='right')]
            assert history[-1]==p.origin
            p.diagnostics['recent_spacing_s']=np.diff(history[-9:]).mean() if len(history)>1 else 0.
            for accum,v in zip(blocks,(*rows(p,r['features']['P']),r['truth'][50*HS]),strict=True):accum.append(v)
            labels.extend([(route,j,r['group'],h) for h in HS])
    return [np.concatenate(a) for a in blocks],labels

def target_radius(c):return np.linalg.norm(c,axis=-1)
def loss_angle(p,t):return np.linalg.norm(angle_error(angles(p),angles(t)),axis=-1)
def grid(imm,route,truth):
    choices=imm[:,None]+ALPHA[None,:,None]*(route-imm)[:,None]
    return loss_angle(choices,truth[:,None])
def correction(rc,t):
    return np.column_stack([angle_error(angles(t),angles(rc)),target_radius(t)-target_radius(rc)])
def fit(data):
    x,i,p,r,t=data
    def tree():return LGBMRegressor(n_estimators=150,learning_rate=.04,num_leaves=15,min_child_samples=45,n_jobs=1,random_state=7,verbosity=-1)
    losses=grid(i,p,t)
    # Under angular loss the useful supervision vanishes when both experts coincide.
    w=np.minimum(loss_angle(i,p),np.deg2rad(15))**2
    g=tree().fit(x,ALPHA[losses.argmin(1)],sample_weight=w/max(w.mean(),1e-12)).booster_
    y=correction(r,t)
    caps=np.stack([np.quantile(abs(y[j::4]),.95,axis=0) for j in range(4)])
    clipped=np.clip(y,-np.tile(caps,(len(y)//4,1)),np.tile(caps,(len(y)//4,1)))
    heads=[tree().fit(x,clipped[:,k]).booster_ for k in range(3)]
    return g,heads,caps

def evaluate(data,labels,learned,split,chronos=None,velocity=None):
    x,i,p,r,t=data;g,heads,caps=learned
    wi=np.clip(g.predict(x,num_threads=1),0,1)
    d=np.column_stack([tree.predict(x,num_threads=1) for tree in heads])
    cap=np.tile(caps,(len(x)//4,1))
    d=.5*np.clip(d,-cap,cap)
    a,e=(angles(r)+d[:,:2]).T
    radius=np.maximum(1e-6,target_radius(r)+d[:,2])
    corrected=radius[:,None]*np.column_stack([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    methods={'I':i,'P':p,'RC':r,'新门控':i+wi[:,None]*(p-i),'RC加残差':corrected}
    if chronos is not None:methods['Chronos-2']=chronos
    frames=[]
    for name,c in methods.items():
        out=pd.DataFrame(labels,columns=['unit','uid_or_anchor','group','h'])
        out['split']=split;out['model']=name;out['position_m']=target_radius(c-t)
        out['angle_deg']=np.rad2deg(loss_angle(c,t))
        if velocity is not None:
            speed=target_radius(velocity)
            along=((c-t)*velocity).sum(1)/np.maximum(speed,1e-8)
            out['speed_m_s']=speed;out['along_abs_m']=np.where(speed>=1,abs(along),np.nan)
            out['cross_m']=np.where(speed>=1,np.sqrt(np.maximum(0,out.position_m**2-along**2)),np.nan)
        frames.append(out)
    return pd.concat(frames)

def confirmation():
    engine=Predictor('Chronos-2',7)
    blocks=[[] for _ in range(5)]; cs=[];vel=[];labels=[]
    world_cache.BASE=CACHE
    for w in range(40):
        world=world_cache.paired_world(CFG,'confirm_0db_20261122/v1',w)
        with gzip.open(CACHE/'warm'/f'v1_{w:03}.pkl.gz','rb') as f:trackers=pickle.load(f)
        chor=engine.batch([tr.history for tr in trackers])
        for uid,(tr,(c,_)) in enumerate(zip(trackers,chor,strict=True)):
            p=tr.snapshots[-1]; ts=p.origin+HS
            truth=network.interpolate(world.positions[uid],ts,world.dt)
            vel.append(network.interpolate(world.velocities[uid],ts,world.dt));cs.append(c[IDX])
            for accum,v in zip(blocks,(*rows(p,p.features),truth),strict=True):accum.append(v)
            labels.extend([(w,uid,world.groups[uid],h) for h in HS])
    return [np.concatenate(a) for a in blocks],labels,np.concatenate(cs),np.concatenate(vel)

if __name__=='__main__':
    train,lab=corpus('train')
    assert len(set(r[0] for r in lab))==960
    frozen=fit(train)
    (OUT/'离线候选.pkl').write_bytes(pickle.dumps(frozen))
    validation,lv=corpus('validation');assert len(set(r[0] for r in lv))==400
    a=evaluate(validation,lv,frozen,'留出验证')
    data,lc,ch,v=confirmation()
    b=evaluate(data,lc,frozen,'确认世界预热末',ch,v)
    existing=pd.read_csv(OUT.parent/'RC_Chronos2_差距定位/跨版本归因/门控与专家上限.csv')
    for m,alias in [('RC','RC'),('Chronos-2','tuned')]:
        source=existing[existing.model==alias].sort_values(['world','uid','horizon_s'])
        test=b[b.model==m].sort_values(['unit','uid_or_anchor','h'])
        np.testing.assert_allclose(test.angle_deg,source.angle_deg,rtol=0,atol=2e-4)
    all_rows=pd.concat([a,b],ignore_index=True)
    all_rows.to_csv(OUT/'中心调整逐锚点.csv',index=False)
    by=all_rows.groupby(['split','model','h'])[['position_m','angle_deg']].mean()
    by.to_csv(OUT/'中心调整汇总.csv')
    print(by.round(4))
    q=b[b.model.isin(['RC','Chronos-2']) & (b.h==5)]
    print(q.groupby(['model','group'])[['angle_deg','along_abs_m','cross_m']].mean().round(3))
    print('moving',q.groupby('model').along_abs_m.count(),q.groupby('model')[['along_abs_m','cross_m']].mean())
