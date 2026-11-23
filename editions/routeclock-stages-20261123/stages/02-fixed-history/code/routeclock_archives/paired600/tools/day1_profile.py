"""Cold load and warm GPU timings at the exact five-model input dimensions."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from routeclock.config import dump_json
from routeclock.foundation import ChronosAdapter,tabpfn_regressor

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['chronos','tabpfn']);parser.add_argument('--device',default='cuda')
    parser.add_argument('--checkpoint',default='amazon/chronos-2');parser.add_argument('--repeat',type=int,default=10)
    parser.add_argument('--features',type=int,choices=[16,63],default=16);parser.add_argument('--out',type=Path,default=ROOT/'runs/day1')
    args=parser.parse_args();cuda=args.device.startswith('cuda');rng=np.random.default_rng(7)
    def sync():
        if cuda:torch.cuda.synchronize(args.device)
    def timed(fn):
        sync();t=time.perf_counter();v=fn();sync();return v,time.perf_counter()-t
    if cuda:torch.cuda.reset_peak_memory_stats(args.device)
    init=time.perf_counter()
    if args.kind=='tabpfn':
        X=rng.normal(size=(7200,args.features));y=rng.normal(size=7200)
        model=tabpfn_regressor(args.device);model.fit(X,y)
        query=X[:16].copy();cases={'query16':lambda:model.predict(query)}
    else:
        model=ChronosAdapter(args.device,args.checkpoint);n=720
        names=['measurement_mask','measurement_age','radial_velocity','radial_std',
               *[f'position_cov_{i}{j}' for i,j in zip(*np.triu_indices(3))],*[f'position_{i}' for i in range(3)]]
        past={k:rng.normal(size=n).astype('float32') for k in names}
        target=rng.normal(size=(3,n)).astype('float32')
        keys=['route_x','route_y','route_z','route_std_x','route_std_y','route_std_z','route_support']
        known={k:rng.normal(size=20).astype('float32') for k in keys}
        plain={'target':target,'past_covariates':past}
        route=dict(target=target,past_covariates={**past,**{k:rng.normal(size=n).astype('float32') for k in keys}},future_covariates=known)
        cases={k:(lambda data=v:model.predict_inputs([(data,np.zeros(3),np.eye(3))])) for k,v in [('plain',plain),('with_route',route)]}
    sync();report=dict(device=args.device,hardware=torch.cuda.get_device_name(args.device) if cuda else 'cpu',
            model=args.kind,init_and_fit_s=time.perf_counter()-init,timings={})
    for k,fn in cases.items():
        _,first=timed(fn);samples=[timed(fn)[1] for _ in range(args.repeat)]
        report['timings'][k]=dict(first_s=first,samples_s=samples,median_s=np.median(samples),p95_s=np.quantile(samples,.95))
    report['peak_allocated_GiB']=torch.cuda.max_memory_allocated(args.device)/2**30 if cuda else None
    report['peak_reserved_GiB']=torch.cuda.max_memory_reserved(args.device)/2**30 if cuda else None
    dump_json(args.out/f'{args.kind}_{args.features}.json',report)
if __name__=='__main__':main()

