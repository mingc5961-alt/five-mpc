"""Audit pilot tensors, compare frozen variants, and replay formal test clocks."""
import argparse
import json
import pickle
from tempfile import TemporaryDirectory
from pathlib import Path
from pilot_histories import ROOT, OUT, configuration
import numpy as np
import pandas as pd
from routeclock.foundation import ChronosAdapter
from routeclock.dataset import observation_schedule, center_metrics, generate_anchors
from routeclock.world import rng_for


def replay_test_clocks(cfg):
    rows=[]
    for mission in range(cfg['data']['single_uav_tasks']['test']):
        rng=rng_for('offline/test',mission,777)
        anchors=[rng.uniform(a,b) for a,b in cfg['data']['anchor_intervals_s']]
        protocol=mission%6
        times=np.array(observation_schedule(anchors,anchors,[.5,1,2,5,10,None][protocol],rng))
        for origin in anchors:
            observed=times[times<=origin]
            n=min(720,int((origin-observed[0])/.5)+1)
            grid=origin-.5*np.arange(n-1,-1,-1)
            idx=np.searchsorted(observed,grid,side='right')-1
            missing=(idx<0)|(grid-observed[np.maximum(idx,0)]>10)
            rows.append(dict(mission=mission,origin=origin,protocol=protocol,rows=n,
                             missing_snapshot_rows=int(missing.sum())))
    return pd.DataFrame(rows)


class AuditAdapter:
    def __init__(self,model):
        self.model=model
        self.snapshots={}

    def predict(self,history,snapshots,p,pcov,support,with_route):
        inp,offset=self.model.inputs(history,snapshots,p,np.sqrt(np.maximum(0,np.diagonal(pcov,axis1=1,axis2=2))),support,with_route)
        item=(inp,offset,history[-1].covariance)
        original=self.model.predict_inputs([item],len(p)-1)[0]
        if with_route:
            masked=self.model.with_route_mask(inp)
            result=self.model.predict_inputs([(masked,*item[1:])],len(p)-1)[0]
            masks=[masked[key]['route_mask'] for key in ('past_covariates','future_covariates')]
            self.snapshots[history[-1].t]=(result[0],[(len(m),int((m==0).sum())) for m in masks])
        return original


def formal_shard(cfg,model,out,start,stop):
    if not 0<=start<stop<=cfg['data']['single_uav_tasks']['test']:
        raise ValueError('Invalid test route range')
    with TemporaryDirectory(dir=out) as tmp:
        for mission in range(start,stop):
            adapter=AuditAdapter(model)
            anchors=generate_anchors(cfg,'test',1,Path(tmp)/'anchors.pkl',adapter,start=mission,anchor_chronos=True)
            counts=[]
            for a in anchors:
                pkg=a['package']
                pkg.centers['route_mask'],masks=adapter.snapshots[pkg.origin]
                for channel,(n,k) in zip(('past','future'),masks,strict=True):
                    counts.append(dict(task=a['task'],origin=pkg.origin,channel=channel,protocol=mission%6,rows=n,missing_rows=k))
            pd.DataFrame(counts).to_csv(out/f'formal_inputs_{mission:04}.csv',index=False)
            pd.DataFrame(center_metrics(anchors,['C','CR_raw','route_mask'])).to_csv(out/f'formal_centers_{mission:04}.csv',index=False)


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/v2_review')
    parser.add_argument('--formal-start',type=int)
    parser.add_argument('--formal-stop',type=int,default=400)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--checkpoint')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    replay_test_clocks(configuration()).to_csv(args.out/'formal_clock_audit.csv',index=False)
    torch.set_num_threads(2)
    manifest=json.loads((OUT/'results/manifest.json').read_text())
    model=ChronosAdapter(args.device,args.checkpoint or str((ROOT/manifest['checkpoint']).resolve()))
    if args.formal_start is not None:
        formal_shard(configuration(),model,args.out,args.formal_start,args.formal_stop)
        return
    anchors=[]
    for path in sorted(OUT.glob('network_*.pkl')):
        with path.open('rb') as stream:
            anchors.extend(pickle.load(stream)[0])
    counts=[]
    for a in anchors:
        inp=a['inputs'][1][0]
        for channel in ('past_covariates','future_covariates'):
            vals=np.column_stack([inp[channel][k] for k in inp['future_covariates']])
            counts.append(dict(task=a['task'],channel=channel,rows=len(vals),
                               missing_rows=int((~np.isfinite(vals)).any(axis=1).sum())))
    pd.DataFrame(counts).to_csv(args.out/'pilot_input_audit.csv',index=False)
    for name,k in [('C',0),('CR_raw',1),('route_mask',1)]:
        for start in range(0,len(anchors),4):
            batch=anchors[start:start+4]
            items=[]
            for a in batch:
                inp,offset=a['inputs'][k]
                if name=='route_mask':
                    inp=model.with_route_mask(inp)
                past=inp['past_covariates']
                cov=np.array([[past[f'position_cov_{min(i,j)}{max(i,j)}'][-1] for j in range(3)] for i in range(3)])
                items.append((inp,offset,cov))
            for a,(pos,_) in zip(batch,model.predict_inputs(items),strict=True):
                a['package'].centers[name]=pos
    df=pd.DataFrame(center_metrics(anchors,['C','CR_raw','route_mask']))
    df.to_csv(args.out/'pilot_forecasts.csv',index=False)
    print(df.query('horizon==10').groupby('method')[['ADE_m','FDE_m','angular_RMSE_deg']].mean())


if __name__=='__main__':
    main()
