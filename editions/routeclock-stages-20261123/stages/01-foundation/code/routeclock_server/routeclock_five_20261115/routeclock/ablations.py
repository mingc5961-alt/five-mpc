"""Executable A2/A3 diagnostics and retrained evidence/gate ablations."""
import copy
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .training import fit_scale,fit_oof_gates,action_labels
from .validity import QUERY_H,interval_errors
from .radio import Radio
from .scheduler import frontier,choose_actions
from .geometry import angles,angle_error
from .config import dump_json

def mutate_evidence(splits,mode,seed):
    result=copy.deepcopy(splits);rng=np.random.default_rng(seed)
    for split in result:
        for center in split[0]['features']:
            if mode=='shuffle':
                local=np.array([a['features'][center][0,16:28] for a in split]);local=(local-local.mean(0))/np.maximum(local.std(0),1e-5)
                near=NearestNeighbors(n_neighbors=min(20,len(split))).fit(local).kneighbors(local,return_distance=False)
                frozen=[a['features'][center].copy() for a in split]
                for j,a in enumerate(split):
                    options=[k for k in near[j] if split[k]['task']!=a['task']]
                    if not options:raise ValueError('Shuffle requires multiple independent task donors')
                    donor=frozen[int(rng.choice(options))];a['features'][center][:,1:16]=donor[:,1:16];a['features'][center][:,28:]=donor[:,28:]
            elif mode=='no_path':
                for a in split:a['features'][center][:,[1,2,3,4,5,13,14,15]]=0;a['features'][center][:,56:]=0
            elif mode=='no_archive':
                for a in split:a['features'][center][:,6:12]=0;a['features'][center][:,28:56]=0
            else:raise ValueError(mode)
    return result

def fixed_center_ablation(splits,center,kind,device,seed,mode,out):
    # Center arrays are left untouched; only the evidence representation changes.
    transformed=mutate_evidence(splits,mode,seed);train,val,cal,test=transformed
    head=fit_scale(train,val,cal,center,kind,device,seed);rows=[]
    for a in test:
        e=interval_errors(a['package'],center,a['truth_h'],a['truth']);env=head.raw(a['package'],a['features'][center])*head.factor
        rows.append({'task':a['task'],'ablation':mode,'center':center,'covered':bool(np.all(e<=env)),'halfwidth_deg':float(np.rad2deg(env).mean())})
    dump_json(out,rows);return rows

def action_diagnostics(anchors,bundle,cfg,out):
    radio=Radio(cfg);rows=[]
    for center in ['RC','CR']:
        if center not in bundle['heads']:continue
        variants={'full':bundle['heads'][center],**{k:v for k,v in bundle['controls'][center].items() if k in ['global','local','no_archive_fixed_center']}}
        variants['direct_duration']=bundle['heads'][center]
        for a in anchors:
            pkg=a['package'];features=a['features'][center]
            for name,head in variants.items():
                env=head.predict(pkg,features);direct=bundle['controls'][center]['duration'] if name=='direct_duration' else None
                fs=frontier(pkg,center,env,radio,cfg,direct,features);action=choose_actions([fs],cfg['simulation']['observation_cost_s'],cfg['simulation']['sensing_budget'])[0]
                if action is None:
                    rows.append({'task':a['task'],'center':center,'head':name,'interval':0.,'net_mbps':0.,'expired':True});continue
                delta=action.interval;mask=a['truth_h']<=delta;pred=np.stack([pkg.at(center,float(h)) for h in a['truth_h'][mask]])
                snr=radio.snr(a['truth'][mask],angles(pred),action.pair,a['shadow'][mask]);expected=radio.snr(pred,angles(pred),action.pair)
                rate=np.array([radio.goodput(s,radio.mcs(est))[0] for s,est in zip(snr,expected)])
                cost=cfg['simulation']['observation_cost_s'];bits=np.trapezoid(rate,a['truth_h'][mask])*max(0,1-cost/delta)
                first_loss=action_labels(a,center,action.pair,radio,'duration')
                rows.append({'task':a['task'],'center':center,'head':name,'interval':delta,'pair':action.pair,'first_loss_s':first_loss,'net_mbps':float(bits/delta/1e6),'expired':bool(first_loss<delta)})
    dump_json(out,rows);return rows

def residual_gate_ablation(splits,seed,out):
    from .dataset import center_metrics
    train,val,cal,test=copy.deepcopy(splits);_,gates=fit_oof_gates(train,val,cal,test,seed,gate_mode='residual')
    rows=center_metrics(test,list(gates));dump_json(out,rows);return rows
