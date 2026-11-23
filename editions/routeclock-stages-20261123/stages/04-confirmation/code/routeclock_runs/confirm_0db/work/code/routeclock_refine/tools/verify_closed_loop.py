"""Implementation acceptance, not an algorithm-ranking experiment."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from routeclock.training import load_bundle
from routeclock.world import World
from routeclock.simulator import simulate
from routeclock.foundation import ChronosAdapter
from routeclock.config import dump_json
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--methods',default='I,C,CR,RC');p.add_argument('--fixed-source');p.add_argument('--device',default='cpu');p.add_argument('--controls',action='store_true');p.add_argument('--uavs',type=int);a=p.parse_args()
torch.set_num_threads(4);bundle=load_bundle(a.bundle);cfg=bundle['config'];methods=a.methods.split(',')
if a.uavs is not None:cfg['simulation']['primary_N']=a.uavs
if a.fixed_source:
    bundle['fixed']=load_bundle(a.fixed_source)['fixed']
    if bundle['fixed'] is None:raise ValueError('Source has no validation-selected fixed policy')
adapter=ChronosAdapter(a.device) if set(methods)&{'C','CR'} else None
world=World(cfg,'network/final_code_acceptance',31000);results=[]
jobs=[(m,None,'value','joint') for m in methods]
if a.controls:
    jobs += [('RC','no_archive','value','joint'),('RC','duration','value','joint'),('RC',None,'edf','joint'),('RC',None,'uniform','joint'),('RC',None,'value','interval_only'),('RC',None,'value','width_only')]
for k,(m,control,scheduler,mode) in enumerate(jobs):
    path=Path(a.out)/f'{k:02d}_{m}_{control or mode}_{scheduler}'
    result=simulate(cfg,world,bundle,m,adapter,path,control,scheduler,mode)
    events=json.loads((path/'events.json').read_text(encoding='utf-8'));pred=json.loads((path/'predictions.json').read_text(encoding='utf-8'))
    actual_bits=sum(e.get('delivered_bits',0) for e in events)
    assert np.isclose(actual_bits/result['service_s']/1e6,result['net_mbps'],rtol=1e-9,atol=1e-9)
    assert np.isclose(result['accounted_s'],result['service_s'])
    assert all(x['origin']==e['origin'] for x,e in zip(pred,[e for e in events if e['type']=='observation']))
    assert all(e.get('prediction_age',0)<=10+1e-8 for e in events)
    result['event_bits_verified']=True;results.append(result)
    dump_json(Path(a.out)/'verification.json',results)
    print({k:result[k] for k in ['method','head','control','scheduler','action_mode','net_mbps','observations_total','event_bits_verified']},flush=True)
