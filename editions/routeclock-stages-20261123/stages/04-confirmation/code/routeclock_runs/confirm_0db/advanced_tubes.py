"""Frozen-center prediction and calibration shared by offline and closed-loop runs."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import torch
from torch import nn
from lightgbm import LGBMRegressor
from chronos import Chronos2Pipeline
from train_vector_long import Center, ROOT, features as velocity_features
from train_chronos_long import CHECKPOINT, context, split_data
from train_raft_long import Model, Bank, route_ids
from refine import tubes
from refine.samples import paths, records
from refine.models import forecast
from routeclock.geometry import angular_jacobian
from routeclock.validity import EPS, finite_quantile

class Predictor:
    def __init__(self,name,seed):
        self.name=name
        folder=ROOT/'training'/name/str(seed)
        self.sha=json.loads((folder/'complete.json').read_text())['sha256']
        with (folder/'best.pt').open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==self.sha
        if name=='Chronos-2':
            self.net=Chronos2Pipeline.from_pretrained(str(CHECKPOINT),device_map='cuda',dtype=torch.float32).model
        elif name=='VECTOR':
            self.net=Center().cuda()
            self.net.gru.fc.register_forward_pre_hook(self.capture)
        elif name=='RAFT':
            self.net=Model().cuda()
            self.net.linear_pred.register_forward_pre_hook(self.capture)
            self.bank=Bank(split_data('train'),route_ids())
        else:
            raise ValueError(name)
        self.net.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True))
        self.net.eval()
        self.head=None

    def capture(self,module,args):
        self.rep=args[0]

    @torch.no_grad()
    def raw(self,histories):
        groups=[]
        for h in histories:
            t=np.asarray([o.t for o in h])
            p=np.stack([o.position for o in h])
            if self.name=='VECTOR':
                groups.append(velocity_features(t,p,t[-1]))
            else:
                groups.append(context(t,p,t[-1])[:2])
        origin=torch.as_tensor(np.stack([h[-1].position for h in histories]),device='cuda',dtype=torch.float32)
        if self.name=='VECTOR':
            x=torch.as_tensor(np.stack(groups),device='cuda',dtype=torch.float32)
            centers=self.net(x)+origin[:,None]
            rep=self.rep
        else:
            x,mask=(torch.as_tensor(np.stack(a),device='cuda',dtype=torch.float32) for a in zip(*groups))
            n=len(x)
            if self.name=='Chronos-2':
                out=self.net(context=x.transpose(1,2).reshape(-1,360),context_mask=mask.repeat_interleave(3,0),group_ids=torch.arange(n,device='cuda').repeat_interleave(3),num_output_patches=2)
                ids=[list(self.net.chronos_config.quantiles).index(q) for q in (.1,.5,.9)]
                q=out.quantile_preds[:,ids,:20].reshape(n,3,3,20).permute(0,3,1,2).cpu().numpy()
                return [(*forecast(v+h[-1].position[None,:,None],h[-1]),None) for h,v in zip(histories,q)]
            neighbors=self.bank.retrieve(x,mask,torch.full((n,),-1,device='cuda'))
            centers=self.net(x,neighbors)+origin[:,None]
            rep=self.rep.reshape(n,3,2,20).permute(0,3,1,2).reshape(n,20,6)
        preds=centers.cpu().numpy()
        reps=rep.cpu().numpy()
        if self.head is not None:
            widths=self.head(rep).cpu().numpy()
        else:
            widths=np.ones((len(preds),20,2))
        return [(np.vstack([h[-1].position,p]),np.maximum(EPS,np.vstack([abs(angular_jacobian(h[-1].position))@np.sqrt(np.maximum(0,np.diag(h[-1].covariance))),w])),r) for h,p,w,r in zip(histories,preds,widths,reps)]

    def batch(self,histories):
        return [(p,w) for p,w,_ in self.raw(histories)]

def new_head(dim):
    return nn.Sequential(nn.Linear(dim,32),nn.SiLU(),nn.Linear(32,2),nn.Softplus()).cuda()

def collect(net,files):
    pending=[]
    rows=[]
    def flush():
        for (h,y),(p,w,r) in zip(pending,net.raw([h for h,_ in pending]),strict=True):
            rows.append((tubes.features(h,p),tubes.errors(p,y),w,r))
        pending.clear()
    for f in files:
        for record in records(f):
            pending.append(record)
            if len(pending)==24: flush()
    if pending: flush()
    return tuple(None if col[0] is None else np.stack(col) for col in zip(*rows))

def fit(net,seed,folder):
    torch.manual_seed(seed)
    train=[p for p in paths('train') if int(p.stem)%5==0]
    assert len(train)==240
    x,err,_,rep=collect(net,train)
    target=err[:,::25]
    bundle={'fixed':np.maximum(EPS,np.quantile(target,.9,axis=0)),'factors':{},'center_sha256':net.sha}
    tree=LGBMRegressor(objective='quantile',alpha=.9,n_estimators=130,num_leaves=15,n_jobs=1,random_state=seed,verbosity=-1)
    tree.fit(x.reshape(-1,x.shape[-1]),target.reshape(-1))
    bundle['tree']=tree
    if rep is not None:
        dim=rep.shape[-1]
        net.head=new_head(dim)
        # Predict angular half-widths directly in radians.
        with torch.no_grad():
            net.head[2].bias.fill_(-3.)
        opt=torch.optim.Adam(net.head.parameters(),lr=.001)
        u=torch.as_tensor(rep,device='cuda')
        y=torch.as_tensor(target[:,1:],dtype=torch.float32,device='cuda')
        for _ in range(12):
            for ids in torch.randperm(len(y),device='cuda').split(48):
                residual=y[ids]-net.head(u[ids])
                loss=torch.maximum(.9*residual,-.1*residual).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        bundle['head']=(dim,{k:v.detach().cpu() for k,v in net.head.state_dict().items()})
    reports={}
    for split in ['calibration','validation']:
        files=paths(split)
        assert len(files)==400
        f,e,native,_=collect(net,files)
        widths={'native':native,'fixed':np.broadcast_to(bundle['fixed'],native.shape),'lightgbm':np.maximum(EPS,tree.predict(f.reshape(-1,f.shape[-1]),num_threads=1).reshape(native.shape))}
        for method,w in widths.items():
            scores=tubes.scores(e,w)
            if split=='calibration':
                assert len(scores)==400
                bundle['factors'][method]=np.maximum.accumulate([finite_quantile(v,.1) for v in scores.T])
            else:
                factors=bundle['factors'][method]
                reports[method]=dict(coverage=(scores<=factors).mean(0).tolist(),width=float(np.mean(w[:,1:])*np.mean(factors)))
    # Keep a narrow candidate even if coverage ranks below the other candidates.
    tight=min(reports,key=lambda k:(reports[k]['width'],k))
    other=min((k for k in reports if k!=tight),key=lambda k:(max(0,.9-min(reports[k]['coverage'])),reports[k]['width'],k))
    bundle['candidates']=[tight,other]
    folder.mkdir(parents=True,exist_ok=True)
    temp=folder/'bundle.tmp'
    with temp.open('wb') as fp:pickle.dump(bundle,fp)
    temp.replace(folder/'bundle.pkl')
    (folder/'offline.json').write_text(json.dumps(dict(seed=seed,model=net.name,validation=reports,candidates=bundle['candidates'],center_sha256=net.sha),indent=2))
    return bundle

def get_bundle(net,seed):
    folder=ROOT/'tubes'/net.name/str(seed)
    if not (folder/'bundle.pkl').exists(): return fit(net,seed,folder)
    with (folder/'bundle.pkl').open('rb') as f: bundle=pickle.load(f)
    assert bundle['center_sha256']==net.sha
    if 'head' in bundle:
        dim,weights=bundle['head'];net.head=new_head(dim);net.head.load_state_dict(weights)
    return bundle
