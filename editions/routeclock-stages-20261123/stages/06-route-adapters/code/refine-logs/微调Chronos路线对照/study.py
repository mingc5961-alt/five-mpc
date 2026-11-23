from pathlib import Path
import sys
import hashlib
import pickle
import json
import datetime
import os
import numpy as np

OUT = Path(__file__).resolve().parent
SPECIAL = OUT.parent / '专用尺度管实验'
sys.path.insert(0, str(SPECIAL))
from protocol import ROOT, PRIOR, write
import features
from features import K, common
from fit_heads import predict, inputs
from refine.data import ARCHIVE
from refine.samples import paths, records
from refine import tubes, network, world_cache
from refine.models import forecast, HS
from routeclock.geometry import angles, angle_error
from routeclock.validity import EPS, QUERY_H, finite_quantile
from routeclock.horizon_gate import HorizonGate
from routeclock.config import load_config
from runtime import remaining_runs as runtime
from train_chronos_long import context
from advanced_tubes import Predictor
import torch

SOURCE = SPECIAL / 'heads/Chronos-2/special.pkl'
CFG = json.loads((SPECIAL / '复现实验配置.json').read_text())
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == CFG['head_sha256']['Chronos-2']['special']
ORIGINAL = pickle.loads(SOURCE.read_bytes())
ARMS = ['C-native','C-special','CR-transfer','CR-special','CRG-special','RC-special']
def log(msg):
    stamp = datetime.datetime.now().astimezone().strftime('%F %T %Z')
    with (OUT.parent / '实时实验进度.txt').open('a') as f:
        print(stamp,'微调Chronos路线对照',msg,file=f)
    print(stamp,msg,flush=True)

def route_inputs(pkg, obs):
    assert pkg.origin == obs.t
    std = np.sqrt(np.maximum(np.diagonal(pkg.covariance['P'],axis1=1,axis2=2),0))
    return np.column_stack([pkg.centers['P'][1:]-obs.position, std[1:],pkg.diagnostics['support'][1:]])

def private_features(q,c,width):
    asym=np.column_stack([q[:,:,1]-q[:,:,0],q[:,:,2]-q[:,:,1]])
    growth=np.vstack([np.zeros(2),np.diff(width,axis=0)/.5])
    return np.column_stack([np.vstack([np.zeros(6),asym])[K],width[K],np.maximum.accumulate(width,axis=0)[K],growth[K]])

class Routed:
    def __init__(self):
        self.engine=Predictor('Chronos-2',7)
        assert self.engine.sha==CFG['center_sha256']['Chronos-2']
        self.ids=[self.engine.net.chronos_config.quantiles.index(q) for q in (.1,.5,.9)]
    @torch.inference_mode()
    def batch(self,histories,packages):
        n=len(histories)
        # Seven known-future covariates from the causal route expert at issuance.
        # No artificial observations are placed in the covariates' past.
        x=np.zeros((n,10,360),np.float32)
        mask=np.zeros_like(x)
        fut=np.zeros((n,10,20),np.float32)
        fm=np.zeros_like(fut)
        for i,(h,p) in enumerate(zip(histories,packages,strict=True)):
            t=np.array([o.t for o in h]); xyz=np.stack([o.position for o in h])
            value,valid,_=context(t,xyz,t[-1])
            x[i,:3]=value.T;mask[i,:3]=valid
            fut[i,3:]=route_inputs(p,h[-1]).T; fm[i,3:]=1
        if not np.isfinite(fut).all():raise ValueError('nonfinite causal route forecast')
        def tensor(a):return torch.as_tensor(a.reshape(-1,a.shape[-1]),device='cuda')
        out=self.engine.net(context=tensor(x),context_mask=tensor(mask),group_ids=torch.arange(n,device='cuda').repeat_interleave(10),
                            future_covariates=tensor(fut),future_covariates_mask=tensor(fm),num_output_patches=2)
        q=out.quantile_preds.reshape(n,10,-1,32)[:,:3,self.ids,:20].permute(0,3,1,2).cpu().numpy()
        rows=[]
        for value,h in zip(q,histories,strict=True):
            absolute=value+h[-1].position[None,:,None]; c,w=forecast(absolute,h[-1])
            rows.append((c,private_features(absolute,c,w)))
        return rows

def gate_x(pkg,base,h):
    times=[o.t for o in h[-9:]]
    pkg.diagnostics['recent_spacing_s']=float(np.mean(np.diff(times))) if len(times)>1 else 0.
    return HorizonGate('g1',7).inputs(base,pkg)

def offline(split):
    dest=OUT/(split+'.npz')
    if dest.exists():return
    model=Routed(); rows=[];pending=[]
    def flush():
        if not pending:return
        pred=model.batch([v[2] for v in pending],[v[3]['package'] for v in pending])
        for (uid,idx,h,r,truth),(c,z) in zip(pending,pred,strict=True):
            p=r['package']; gx=gate_x(p,r['features']['P'],h)
            rows.append((uid,idx,c,z,common(h,c),gx,p.centers['I'],truth,r['sample_h']))
        pending.clear()
    for file in paths(split):
        archived=pickle.loads((ARCHIVE/split/(file.stem+'.pkl')).read_bytes())
        for j,((h,truth),r) in enumerate(zip(records(file),archived,strict=True)):
            if not np.array_equal(truth,r['truth']):raise ValueError('unaligned truth')
            pending.append((int(file.stem),j,h,r,truth))
            if len(pending)==8:flush()
    flush()
    fields=('ids','anchor','center','special','common','gate','imm','truth','sample_h')
    cols=dict(zip(fields,map(np.stack,zip(*rows,strict=True)),strict=True))
    assert np.isfinite(cols['center']).all() and len(rows)=={'train':3600,'validation':1200,'calibration':400}[split]
    tmp=dest.with_suffix('.tmp.npz');np.savez(tmp,**cols);tmp.replace(dest)
    log(f'{split} 因果路线协变量推理完成，{len(set(cols["ids"]))} 条路线，{len(rows)} 个历史起点，来源逐条核对。')

if __name__=='__main__' and sys.argv[1]=='offline':offline(sys.argv[2])

def fused_values(gate,gx,c,imm):
    w=gate.weights(gx)
    return w[:,None]*c+(1-w[:,None])*imm,w

def make_head(arm, split, data, gate):
    rows=[]; selects=(data['ids']%5==0) if split=='train' else np.ones(len(data['ids']),bool)
    for i in np.where(selects)[0]:
        c=data['center'][i]; cx=data['common'][i]; z=data['special'][i]
        if arm.startswith('CRG'):
            c,w=fused_values(gate,data['gate'][i],c,data['imm'][i])
            # Recompute geometry while retaining the same observation-only columns.
            obs=cx[:,:15]
            drift=angle_error(angles(c),angles(c[0]))
            cx=np.column_stack([obs,c[K]-c[0],np.linalg.norm(c[K],axis=1),drift[K],np.maximum.accumulate(abs(drift),axis=0)[K]])
            if arm=='CRG-special':
                z=np.column_stack([z,w[K],data['center'][i,K]-data['imm'][i,K]])
        rows.append((data['ids'][i],cx,z,tubes.errors(c,data['truth'][i])))
    return dict(zip(('ids','common','special','error'),map(np.stack,zip(*rows,strict=True)),strict=True))

def refit(label,groups):
    from lightgbm import LGBMRegressor,early_stopping
    d=groups['train']; x=inputs(d['common'],d['special'],None).reshape(len(d['ids']),len(K)*2,-1)
    y=np.log(EPS+np.maximum.accumulate(d['error'],axis=1)[:,np.rint(QUERY_H/.02).astype(int)]).reshape(len(x),-1)
    val=d['ids']%25==0
    def design(sel):return x[sel].reshape(-1,x.shape[-1]),y[sel].ravel()
    scores=[]
    for leaves,child in [(15,20),(15,50),(31,20),(31,50)]:
        reg=LGBMRegressor(num_leaves=leaves,min_child_samples=child,n_estimators=500,n_jobs=1,random_state=7,verbosity=-1)
        reg.fit(*design(~val),eval_set=[design(val)],callbacks=[early_stopping(30,verbose=False)])
        xv,yv=design(val); score=np.mean((reg.booster_.predict(xv,num_threads=1)-yv)**2)
        scores.append((score,leaves,child,reg.best_iteration_))
    _,leaves,child,n=min(scores)
    reg=LGBMRegressor(num_leaves=leaves,min_child_samples=child,n_estimators=n,n_jobs=1,random_state=7,verbosity=-1)
    reg.fit(*design(np.ones(len(x),bool)))
    head={'booster':reg.booster_,'pca':None,'kind':'special','model':label,'params':[leaves,child,n]}
    cal=groups['calibration'];s=tubes.scores(cal['error'],predict(head,cal['common'],cal['special']))
    head['factors']=np.maximum.accumulate([finite_quantile(col,.1) for col in s.T])
    return head

def nat_width(z):
    # Native width at the seven nonzero deadlines is insufficient for control.
    # Recovered from 20-step q10/q90 in online inference. Stored below for calibration.
    return z

def fit_all():
    data={s:dict(np.load(OUT/(s+'.npz'))) for s in ('train','calibration','validation')}
    def arr(d,sel):
        rows=np.flatnonzero(sel); h=d['sample_h'][rows]; j=np.rint(h*2).astype(int)
        return d['gate'][rows,j],d['center'][rows,j],d['imm'][rows,j],d['truth'][rows,np.rint(h*50).astype(int)]
    train=data['train'];valid=data['validation']
    a=arr(train,train['ids']%5!=0);v=arr(valid,np.ones(len(valid['ids']),bool))
    choices=[]
    for l2 in (.0001,.001,.01):
        g=HorizonGate('g1',7).fit(*a,l2=l2)
        diff=angle_error(angles(g.predict(*v[:3])),angles(v[3]))
        choices.append((np.mean(g.importance(v[0])*np.sum(diff**2,axis=1)),l2,g))
    loss,l2,g=min(choices,key=lambda row:row[0])
    assets={'gate':g,'gate_l2':l2}
    reports={}
    for label in ('CR-special','CRG-special','CR-transfer','CRG-transfer'):
        groups={s:make_head(label,s,d,g) for s,d in data.items()}
        if label in ('CR-special','CRG-special'): head=refit(label,groups)
        else:head=ORIGINAL if label=='CR-transfer' else assets['CR-special']
        assets[label]=head
        d=groups['validation'];w=predict(head,d['common'],d['special'])
        s=tubes.scores(d['error'],w)
        reports[label]={'coverage':(s<=head['factors']).mean(0).tolist(),'halfwidth_deg':np.rad2deg(w[:,K[1:]]*head['factors'][None,:,None]).mean((0,2)).tolist()}
    # Native calibration uses true per-half-second widths. Re-evaluate calibration in the same grouped-input implementation.
    from unittest.mock import patch
    widths=[]
    import study
    orig=study.private_features
    def collect(q,c,w):
        widths.append(w);return orig(q,c,w)
    eng=Routed(); errors=[];pending=[]
    def flush():
        if not pending:return
        with patch.dict(Routed.batch.__wrapped__.__globals__,{'private_features':collect}):
            preds=eng.batch([r[0] for r in pending],[r[1] for r in pending])
        for (c,_),row in zip(preds,pending,strict=True):errors.append(tubes.errors(c,row[2]))
        pending.clear()
    for file in paths('calibration'):
        r=pickle.loads((ARCHIVE/'calibration'/(file.stem+'.pkl')).read_bytes())[0]
        h,t=next(records(file));pending.append((h,r['package'],t))
        if len(pending)==8:flush()
    flush()
    assert len(widths)==400
    s=tubes.scores(np.stack(errors),np.stack(widths))
    assets['CR-native']=np.maximum.accumulate([finite_quantile(col,.1) for col in s.T])
    path=OUT/'fitted.tmp';path.write_bytes(pickle.dumps(assets));path.replace(OUT/'fitted.pkl')
    write(OUT/'calibration.json',dict(gate_l2=l2,validation=reports,horizons=QUERY_H[1:].tolist()))
    log('路线中心、G1 融合与误差管冻结。G1 用 960 条训练路线，按原规则用验证集选正则，专用管用另 240 条训练路线，400 条独立校准路线；迁移臂完整冻结原树和原校准系数。')

if __name__=='__main__' and sys.argv[1]=='fit':fit_all()
