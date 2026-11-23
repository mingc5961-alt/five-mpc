"""Train on disjoint long histories, evaluate five centers and run paired real-model loops."""
import argparse
import copy
import json
import pickle
import time
import pandas as pd
import numpy as np
from pilot_histories import ROOT,OUT,configuration,event_at
from routeclock.foundation import ChronosAdapter
from routeclock.five import CENTERS,train_five,evaluate_anchors
from routeclock.simulator import simulate
from routeclock.config import dump_json

def archived_gate_events(groups,gate):
    rows=[]
    for _,trackers,world in groups:
        for uid,tracker in enumerate(trackers):
            for pkg in tracker.snapshots:
                if pkg.origin<60:continue
                event,residual=event_at(world,uid,pkg.origin)
                weights=gate.weights(pkg.features[:,:16])
                fused=gate.predict(pkg.features[:,:16],pkg.centers['P'],pkg.centers['I'])
                for h in (1,2,5,10):
                    j=2*h;truth=world.truth(uid,pkg.origin+h)[0]
                    rows.append(dict(mission=world.mission,uid=uid,group=world.groups[uid],origin=pkg.origin,
                          event=event,route_distance_m=residual,horizon=h,weight=weights[j],support=pkg.diagnostics['support'][j],
                          route_m=np.linalg.norm(pkg.centers['P'][j]-truth),imm_m=np.linalg.norm(pkg.centers['I'][j]-truth),
                          gate_m=np.linalg.norm(fused[j]-truth)))
    return pd.DataFrame(rows)

def read(path):
    with path.open('rb') as f:return pickle.load(f)

def main():
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='amazon/chronos-2')
    args=parser.parse_args();torch.set_num_threads(2)
    foundation=ChronosAdapter('cpu',args.checkpoint)
    groups={};sets={}
    expected={'train':12,'validation':6,'cal':12,'network':2}
    for split,count in expected.items():
        paths=sorted(OUT.glob(split+'_*.pkl'))
        if len(paths)!=count:raise ValueError(f'{split}: expected {count} complete files, found {len(paths)}')
        groups[split]=[read(p) for p in paths];sets[split]=[r for records,_,_ in groups[split] for r in records]
    anchors=sum(sets.values(),[])
    hashes=[r['route_hash'] for r in anchors]
    if len(set(hashes))!=len(hashes):raise ValueError('Repeated route geometry')
    for k,method in enumerate(('C','CR_raw')):
        for j in range(0,len(anchors),4):
            batch=anchors[j:j+4]
            items=[]
            for a in batch:
                inp,offset=a['inputs'][k];past=inp['past_covariates']
                cov=np.array([[past[f'position_cov_{min(i,j)}{max(i,j)}'][-1] for j in range(3)] for i in range(3)])
                items.append((inp,offset,cov))
            for a,(pos,cov) in zip(batch,foundation.predict_inputs(items),strict=True):
                a['package'].centers[method]=pos;a['package'].covariance[method]=cov
    def horizons(anchors):
        return [dict(a,sample_h=h) for a in anchors for h in (.5,1.,2.,3.,5.,8.,10.)]
    cfg=configuration();out=OUT/'results'
    b=train_five(horizons(sets['train']),horizons(sets['validation']),sets['cal'],sets['network'],cfg,out,extras=False)
    df=evaluate_anchors(sets['network'],b,out)
    archived_gate_events(groups['network'],b['gates']['RC']).to_csv(out/'historical_gate_events.csv',index=False)
    selected=json.loads((ROOT/'runs/fixed_validation/winner.json').read_text())
    b['fixed']=selected;b['action_reference']=selected
    settings=[(m,'joint') for m in ['F',*CENTERS]]+[('RC',a) for a in ('fixed','width_only','interval_only')]
    rows=[]
    for mission,(_,trackers,world) in enumerate(groups['network']):
        for method,mode in settings:
            path=out/f'loop_{mission}_{method}_{mode}'
            result=simulate(cfg,world,b,method,foundation if method in ('C','CR_raw') else None,path,
                            action_mode=mode,service_trackers=copy.deepcopy(trackers))
            rows.append({k:v for k,v in result.items() if not isinstance(v,(list,dict)) and k!='uav_mbps'})
            pd.DataFrame(rows).to_csv(out/'closed_loop.csv',index=False)
            print(mission,method,mode,round(result['net_mbps'],2),flush=True)
    dump_json(out/'manifest.json',dict(checkpoint=foundation.checkpoint,seed=7,route_hashes=hashes,counts=b['counts'],
              training_horizons=[.5,1,2,3,5,8,10],test_missions=2,observation_history=180,service_s=18,
              heads='LightGBM independently fitted and jointly calibrated, local diagnostics only',fixed=selected))
if __name__=='__main__':main()

