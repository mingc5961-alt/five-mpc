"""Causal adapters and calibration for the six remaining methods."""
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from refine.data import BASE,ROOT,ARCHIVE
from refine import tubes
from refine.models import Chronos,forecast,HS
from refine.samples import records
from routeclock.config import load_config
from routeclock.geometry import angular_jacobian
from routeclock.predictors import Tracker,apply_gate
from routeclock.training import fit_gate
from routeclock.validity import EPS,QUERY_H,finite_quantile
from routeclock.filters import PositionIMM
from routeclock.memory import RouteMemory
from routeclock.evidence import Forecast
from runtime.learned_queue import save,digest

OUT=BASE/'remaining_queue'
ARCHIVED=Path('/data/chen/routeclock_archives/paired600/runs/paired600/models/frozen/bundle.pkl')
INDEX=QUERY_H.searchsorted(HS)
def dumps(path,x):
    tmp=path.with_suffix('.part')
    tmp.write_bytes(pickle.dumps(x))
    tmp.replace(path)

def anchors(split,subset=None):
    for p in sorted((ARCHIVE/split).glob('*.pkl')):
        if subset and not subset(int(p.stem)):continue
        with np.load(BASE/'observations'/split/(p.stem+'.npz')) as data:
            times=data['times']
        for row in pickle.loads(p.read_bytes()):
            pkg=row['package'];h=times[:times.searchsorted(pkg.origin,side='right')]
            assert h[-1]==pkg.origin
            pkg.diagnostics['recent_spacing_s']=float(np.mean(np.diff(h[-9:]))) if len(h)>1 else 0.
            yield row

def factors(errors,raw):
    scores=tubes.scores(np.stack(errors),np.stack(raw))
    assert len(scores)==400 and np.isfinite(scores).all()
    return np.maximum.accumulate([finite_quantile(s,.1) for s in scores.T])

def classic_params():
    b=pickle.loads(ARCHIVED.read_bytes())
    payload={}
    indices={'P':'P','RC':'P','I':'I'}
    fixed=[]
    for r in anchors('train',lambda i:i%5==0):
        p=r['package']
        apply_gate(p,b['gates']['RC'],r['features']['P'],'RC','P')
        fixed.append(tubes.errors(p.centers['RC'],r['truth'])[::25])
    fixed=np.maximum(EPS,np.median(fixed,axis=0))
    errs={m:[] for m in indices};scales={m:[] for m in indices}
    for r in anchors('calibration'):
        p=r['package'];apply_gate(p,b['gates']['RC'],r['features']['P'],'RC','P')
        for m,feature in indices.items():
            raw=b['heads'][m].raw(p,r['features'][feature])[INDEX]
            errs[m].append(tubes.errors(p.centers[m],r['truth']));scales[m].append(raw)
    for m in indices:
        payload[m]=dict(head=b['heads'][m],gate=b['gates'].get(m),factors={'default':factors(errs[m],scales[m])})
    payload['RC'].update(fixed=fixed)
    payload['RC']['factors']['fixed']=factors(errs['RC'],[fixed]*400)
    dumps(OUT/'classical.pkl',payload)

def gate_features(pkg,feat,gate):
    center=gate.predict(gate.inputs(feat,pkg),pkg.centers['CR_raw'],pkg.centers['I'])
    weights=gate.weights(gate.inputs(feat,pkg))
    diag=lambda n:np.sqrt(np.maximum(np.diagonal(pkg.covariance[n],axis1=1,axis2=2),0))
    x=np.column_stack([feat,weights,center,pkg.centers['CR_raw']-pkg.centers['I'],diag('CR_raw'),diag('I')])
    return center,np.column_stack([np.repeat(x,2,axis=0),np.tile([0,1],len(x))])

def gate_params():
    train=list(anchors('train',lambda i:i%5!=0))
    val=list(anchors('validation'))
    gate,l2=fit_gate(train,val,'CR_raw',7,mode='g1')
    del train
    def arrays(rows):
        x=[];y=[]
        for r in rows:
            center,features=gate_features(r['package'],r['features']['CR_raw'],gate)
            x.extend(features);y.extend(np.log(EPS+tubes.errors(center,r['truth'])[::25]).ravel())
        return np.array(x),np.array(y)
    x,y=arrays(anchors('train',lambda i:i%5==0));v,w=arrays(val)
    del val
    best=float('inf');tree=None
    for leaves,minimum in [(15,20),(15,50),(31,20),(31,50)]:
        m=LGBMRegressor(n_estimators=500,num_leaves=leaves,min_child_samples=minimum,n_jobs=1,random_state=7,verbosity=-1)
        m.fit(x,y,eval_set=[(v,w)],callbacks=[early_stopping(30,verbose=False)])
        loss=np.mean((m.predict(v)-w)**2)
        if loss<best:best=loss;tree=m
    errs=[];scales=[]
    for r in anchors('calibration'):
        center,features=gate_features(r['package'],r['features']['CR_raw'],gate)
        errs.append(tubes.errors(center,r['truth']));scales.append(tubes.tree_scale(tree,features[None])[0])
    dumps(OUT/'CR-G.pkl',dict(gate=gate,tree=tree,l2=l2,factors={'default':factors(errs,scales)}))

class RouteQuantiles(Chronos):
    def predict(self,hist,snaps,p):
        cov=p.covariance['P']
        item,offset=self.adapter.inputs(hist,snaps,p.centers['P'],np.sqrt(np.maximum(np.diagonal(cov,axis1=1,axis2=2),0)),p.diagnostics['support'],True)
        qs,_=self.adapter.pipeline.predict_quantiles([item],prediction_length=20,quantile_levels=[.1,.5,.9],cross_learning=False,context_length=720)
        q=qs[0].detach().cpu().numpy()
        center,raw=forecast(q.transpose(1,0,2)+offset[None,:,None],hist[-1])
        # Preserve the same operation order and precision as ChronosAdapter.predict_inputs.
        center=np.vstack([offset,q[:,:,1].T+offset])
        std=(q[:,:,2]-q[:,:,0]).T/2.563103
        cov=np.concatenate([hist[-1].covariance[None],np.array([np.diag(v*v) for v in std])])
        return center,raw,cov

def raw_params():
    cfg=load_config();model=RouteQuantiles();errs=[];scales=[]
    for p in tubes.paths('calibration'):
        for hist,truth in records(p):
            imm=PositionIMM();memory=RouteMemory(cfg['models']['route']);snaps=[]
            for obs in hist:
                imm.observe(obs);memory.update(obs,imm.state()[0][3:])
                center,cov,support,_,_=memory.forecast(HS,False)
                snap=Forecast(obs.t,HS,{'P':center},{'P':cov},None,{'support':support})
                snaps.append(snap)
            center,scale,_=model.predict(hist,snaps,snap)
            errs.append(tubes.errors(center,truth));scales.append(scale)
    dumps(OUT/'CR_raw.pkl',dict(factors={'default':factors(errs,scales)}))

class CausalModel:
    def __init__(self,name,params,cfg,chronos=None):
        self.name=name;self.params=params;self.cfg=cfg
        self.chronos=chronos;self.trackers={}
    def batch(self,histories):
        output=[]
        for h in histories:
            if id(h) not in self.trackers:
                self.trackers[id(h)]=Tracker(self.cfg)
            tr=self.trackers[id(h)]
            for obs in h[len(tr.history):]:
                tr.observe(obs);p=tr.issue()
            p=tr.snapshots[-1]
            if self.name in ('CR-G','CR_raw'):
                center,raw,cov=self.chronos.predict(tr.history,tr.snapshots,p)
                p.centers['CR_raw']=center;p.covariance['CR_raw']=cov
            if self.name=='CR_raw':output.append((center,raw));continue
            if self.name=='CR-G':
                center,x=gate_features(p,tr.features(p,'CR_raw'),self.params['gate'])
                output.append((center,tubes.tree_scale(self.params['tree'],x[None])[0]));continue
            if self.name=='RC':apply_gate(p,self.params['gate'],p.features,'RC','P')
            output.append((p.centers[self.name],self.params['head'].raw(p,p.features)[INDEX]))
        return output

if __name__=='__main__':
    import sys
    OUT.mkdir(exist_ok=True)
    {'classical':classic_params,'CR-G':gate_params,'CR_raw':raw_params}[sys.argv[1]]()
