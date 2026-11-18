import copy,pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from .predictors import Gate,apply_gate
from .validity import ScaleHead,GlobalEnvelope,ActionHead,QUERY_H,feature_rows,interval_errors
from .geometry import angles,angle_error
from .radio import Radio,array_gain
from .config import dump_json

def gate_arrays(anchors,expert):
    X=[]; p=[];i=[];y=[]
    for a in anchors:
        pkg=a['package'];h=a['sample_h'];j=int(round(h/.5))
        X.append(a['features'][expert][j,:16]);p.append(pkg.at(expert,h));i.append(pkg.at('I',h));y.append([np.interp(h,a['truth_h'],a['truth'][:,axis]) for axis in range(3)])
    return tuple(map(np.array,(X,p,i,y)))

def fit_gate(train,val,expert,seed,mask=None):
    arrays=gate_arrays(train,expert);v=gate_arrays(val,expert);best=None
    for l2 in [.0001,.001,.01]:
        g=Gate(seed,mask).fit(*arrays,l2=l2); pred=g.predict(v[0],v[1],v[2]); loss=np.mean(angle_error(angles(pred),angles(v[3]))**2)
        if best is None or loss<best[0]:best=(loss,g,l2)
    return best[1],best[2]

def inject_gate(anchors,gate,name,expert):
    for a in anchors:
        apply_gate(a['package'],gate,a['features'][expert],name,expert); a['features'][name]=a['features'][expert].copy()

def fit_oof_gates(train,val,cal,test,seed=7,gate_mode='full'):
    train=copy.deepcopy(train); gates={}; mask=None
    if gate_mode=='residual':mask=np.zeros(16);mask[[0,1]]=1
    for name,expert in [('RC','P'),('CR','CR_raw')]:
        if expert not in train[0]['package'].centers:continue
        final,l2=fit_gate(train,val,expert,seed,mask);gates[name]=final
        group=np.array([a['task'] for a in train]); splitter=GroupKFold(3)
        for ti,vi in splitter.split(group,groups=group):
            sub=[train[i] for i in ti];g=Gate(seed,mask).fit(*gate_arrays(sub,expert),l2=l2)
            inject_gate([train[i] for i in vi],g,name,expert)
        for split in [val,cal,test]:inject_gate(split,final,name,expert)
    return train,gates

def head_arrays(anchors,center,mode='full'):
    X=[];y=[]
    for a in anchors:
        package=a['package']; h=a['sample_h']; errors=interval_errors(package,center,a['truth_h'],a['truth']); j=int(np.where(QUERY_H==h)[0][0])
        X.extend(feature_rows(a['features'][center],package.hs,[h],mode)); y.extend(errors[j])
    return np.array(X),np.array(y)

def fit_scale(train,val,cal,center,kind,device,seed,mode='full'):
    X,y=head_arrays(train,center,mode);Xv,yv=head_arrays(val,center,mode)
    candidates=[(15,20),(15,50),(31,20),(31,50)] if kind=='lightgbm' else [(15,20)]
    best=None
    for leaves,child in candidates:
        head=ScaleHead(kind,device,seed,mode,{'num_leaves':leaves,'min_child_samples':child}).fit(X,y,Xv,yv)
        score=float(np.mean((head.reg.predict(Xv)-np.log(np.deg2rad(.001)+yv))**2))
        if best is None or score<best[0]:best=(score,head)
    return best[1].calibrate(cal,center)

def action_labels(a,center,pair,radio,target):
    pkg=a['package']; pred=np.stack([pkg.at(center,float(h)) for h in a['truth_h']]); pointing=angles(pred)
    gain=array_gain(pointing,angles(a['truth']),pair); loss=-10*np.log10(gain/np.prod(pair))
    if target=='duration':
        bad=np.flatnonzero(loss>3); return float(a['truth_h'][bad[0]]) if len(bad) else 10.
    snr=radio.snr(a['truth'],pointing,pair,a['shadow']); running=np.minimum.accumulate(snr)
    return np.array([running[min(np.searchsorted(a['truth_h'],h,side='right')-1,len(running)-1)] for h in QUERY_H])

def train_action_head(train,val,cal,center,cfg,kind,device,seed,target):
    head=ActionHead(kind,device,seed,target);radio=Radio(cfg)
    def arrays(anchors):
        X=[];y=[];r=np.random.default_rng(seed)
        for a in anchors:
            for idx in r.choice(9,2,replace=False):
                pair=radio.pairs[idx];h=0 if target=='duration' else a['sample_h']
                X.extend(head.rows(a['features'][center],a['package'].hs,pair,[h]));label=action_labels(a,center,pair,radio,target)
                y.append(label if target=='duration' else label[np.where(QUERY_H==h)[0][0]])
        return np.array(X),np.array(y)
    X,y=arrays(train);Xv,yv=arrays(val);head.fit(X,y,Xv,yv);scores=[]
    for a in cal:
        errors=[]
        for pair in radio.pairs:
            hs=[0] if target=='duration' else QUERY_H
            pred=head.reg.predict(head.rows(a['features'][center],a['package'].hs,pair,hs))
            label=action_labels(a,center,pair,radio,target);errors.append(float(np.max(pred-label)))
        scores.append(max(errors))
    return head.calibrate(scores)

def train_bundle(train,val,cal,test,cfg,out,kind='lightgbm',device='cpu',seed=7,action_heads=True):
    original=copy.deepcopy((train,val,cal,test))
    train,gates=fit_oof_gates(train,val,cal,test,seed)
    centers=['I','P','H','RC']+(['C','CR_raw','CR'] if 'C' in train[0]['package'].centers else [])
    heads={}; controls={}; validity_metrics=[]
    for center in centers:
        mode='local' if center=='C' else 'full'
        print(f'fit {kind} scale {center}',flush=True)
        head=fit_scale(train,val,cal,center,kind,device,seed,mode);heads[center]=head
        for a in test:
            err=interval_errors(a['package'],center,a['truth_h'],a['truth']); env=head.raw(a['package'],a['features'][center])*head.factor
            validity_metrics.append({'task':a['task'],'center':center,'covered':bool(np.all(err<=env)),'mean_halfwidth_deg':float(np.rad2deg(env).mean()),'calibration_factor':head.factor})
    for center in ['RC','CR']:
        if center not in heads:continue
        controls[center]={
            'global':GlobalEnvelope().fit(train,center).calibrate(cal,center),
            'local':fit_scale(train,val,cal,center,kind,device,seed,'local'),
            'no_archive_fixed_center':fit_scale(train,val,cal,center,kind,device,seed,'no_archive')}
        if action_heads:
            for target in ['duration','snr']:
                print(f'fit {target} {center}',flush=True)
                controls[center][target]=train_action_head(train,val,cal,center,cfg,kind,device,seed,target)
    # Closed-loop archive ablation must remove evidence from BOTH gate and head,
    # with refitting. Fixed-center A2 uses the separate head above.
    no_splits=copy.deepcopy(original)
    for split in no_splits:
        for a in split:
            for f in a['features'].values():f[:,6:12]=0;f[:,28:56]=0
    nt,ng=fit_oof_gates(*no_splits,seed)
    for center in ng:
        controls[center]['no_archive_gate']=ng[center]
        controls[center]['no_archive']=fit_scale(nt,no_splits[1],no_splits[2],center,kind,device,seed,'no_archive')
    bundle={'gates':gates,'heads':heads,'controls':controls,'kind':kind,'seed':seed,'device':device,'methods':centers,'config':cfg,'training_tasks':len({a['task'] for a in train}),'calibration_tasks':len(cal),'fixed':None}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'bundle.pkl').open('wb') as f:pickle.dump(bundle,f)
    with (out/'test_predictions.pkl').open('wb') as f:pickle.dump(test,f)
    dump_json(out/'validity_metrics.json',validity_metrics)
    return bundle,test

def load_bundle(path):
    with Path(path).open('rb') as f:return pickle.load(f)
