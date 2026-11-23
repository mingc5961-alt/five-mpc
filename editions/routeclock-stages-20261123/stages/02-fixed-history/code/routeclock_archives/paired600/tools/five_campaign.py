"""Entry point for the five-model campaign; pilot, timing and test namespaces are separate."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from routeclock.config import load_config,dump_json
from routeclock.dataset import generate_anchors,load_anchors
from routeclock.five import CENTERS,evaluate_anchors,train_five
from routeclock.foundation import ChronosAdapter
from routeclock.simulator import simulate
from routeclock.world import World
from routeclock.evaluation import paired_bootstrap

SPLITS=('train','validation','calibration','test')
def jobs(block):
    if block=='main':return [(i,m,'joint',None) for i in range(30) for m in ('F',*CENTERS)]
    if block=='actions':return [(i,'RC',mode,None) for i in range(30) for mode in ('fixed','width_only','interval_only')]
    if block=='reuse':return [(i,m,'joint',p) for i in range(20) for p in (0.,.5,1.) for m in ('RC','I')]
    if block=='repeat':return [(i,m,'joint',None) for i in range(20) for m in ('I','P','RC','C','CR_raw')]
    raise ValueError(block)

def read_bundle(path,device):
    with path.open('rb') as stream:b=pickle.load(stream)
    for head in [*b['heads'].values(),*b.get('controls',{}).get('RC',{}).values()]:
        if getattr(head,'reg',None) is not None:head.reg.device=device
    return b

def run():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['generate','train','network','timing','summary','matrix'])
    p.add_argument('--device',default='cuda');p.add_argument('--checkpoint',default='amazon/chronos-2')
    p.add_argument('--data',type=Path,default=ROOT/'data/five_formal')
    p.add_argument('--out',type=Path,default=ROOT/'runs/five_formal')
    p.add_argument('--bundle',type=Path);p.add_argument('--fixed',type=Path,default=ROOT/'runs/fixed_validation/winner.json')
    p.add_argument('--head',choices=['lightgbm','tabpfn'],default='lightgbm');p.add_argument('--seed',type=int,default=7)
    p.add_argument('--split',choices=SPLITS);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int)
    p.add_argument('--block',choices=['main','actions','reuse','repeat'],default='main')
    args=p.parse_args();cfg=load_config();counts=cfg['data']['single_uav_tasks'];torch.set_num_threads(4)
    dest=args.out;dest.mkdir(parents=True,exist_ok=True)
    if args.stage=='matrix':
        dump_json(dest/'matrix.json',{k:jobs(k) for k in ['main','actions','reuse','repeat']});return
    if args.stage=='generate':
        adapter=ChronosAdapter(args.device,args.checkpoint)
        for split in ([args.split] if args.split else SPLITS):
            for i in range(args.start,min(args.stop if args.stop is not None else counts[split],counts[split])):
                path=args.data/split/f'{i:04}.pkl'
                if not path.exists():generate_anchors(cfg,split,1,path,adapter,start=i,anchor_chronos=True)
        return
    if args.stage=='train':
        splits=[[row for i in range(counts[s]) for row in load_anchors(args.data/s/f'{i:04}.pkl')] for s in SPLITS]
        target=dest/f'{args.head}_{args.seed}'
        b=train_five(*splits,cfg,target,args.head,args.device,args.seed)
        evaluate_anchors(splits[-1],b,target);return
    if args.stage=='summary':
        net=dest/'network'/f'{args.head}_{args.seed}';rows=[]
        for path in sorted(net.glob('*/*/completed.json')):
            r=json.loads(path.read_text());rows.append(r)
        if not rows:raise ValueError('No completed network jobs')
        df=pd.DataFrame(rows);df.to_csv(net/'jobs.csv',index=False);comparisons=[]
        for (block,frac),g in df.groupby(['block','target_reuse'],dropna=False):
            for metric in ['net_mbps','sensing_fraction','narrow_share','narrow_hold_s','hold_failure_fraction','outage_data_fraction']:
                pivot=g.pivot(index='mission',columns='policy',values=metric)
                if block=='actions':
                    ref=df[(df.block=='main')&(df.policy=='RC')].set_index('mission')[metric]
                    pivot['RC']=ref
                if 'RC' not in pivot:continue
                for c in pivot:
                    if c=='RC':continue
                    paired=pivot[['RC',c]].dropna()
                    if len(paired):comparisons.append(dict(block=block,reuse=frac,reference=c,metric=metric,**paired_bootstrap(paired.RC,paired[c])))
        # Decompose the change relative to each baseline using per-mission means.
        decomposition=[]
        for (block,frac),g in df.groupby(['block','target_reuse'],dropna=False):
            if block=='actions':g=pd.concat([g,df[(df.block=='main')&(df.policy=='RC')]])
            ref=g[g.policy=='RC'].set_index('mission')
            for _,r in g[g.policy!='RC'].iterrows():
                if r.mission not in ref.index:continue
                a=ref.loc[r.mission]
                fa=(a.data_s-a.unavailable_s)/a.service_s;fb=(r.data_s-r.unavailable_s)/r.service_s
                if min(fa,fb)<=0:continue
                ra=a.net_mbps/fa;rb=r.net_mbps/fb
                decomposition.append(dict(block=block,reuse=frac,mission=r.mission,reference=r.policy,
                         net=a.net_mbps-r.net_mbps,time=(fa-fb)*(ra+rb)/2,rate=(ra-rb)*(fa+fb)/2))
        pd.DataFrame(decomposition).to_csv(net/'decomposition.csv',index=False)
        pd.DataFrame(comparisons).to_csv(net/'paired.csv',index=False);return
    if args.bundle is None:p.error('--bundle required for timing or network')
    b=read_bundle(args.bundle,args.device)
    b['fixed']=json.loads(args.fixed.read_text(encoding='utf8'));b['action_reference']=b['fixed']
    if args.stage=='timing':
        # Six complete offline missions cover the six sampling protocols and behavior families.
        adapter=ChronosAdapter(args.device,args.checkpoint);bench=dest/'day1';bench.mkdir(exist_ok=True)
        offsets=np.cumsum([0]+cfg['data']['class_counts']['train'][:-1]);timings=[]
        for group,offset in enumerate(offsets):
            i=int(offset+(group-offset)%6);t=time.perf_counter()
            generate_anchors(cfg,'train',1,bench/f'route_{i}.pkl',adapter,start=i,anchor_chronos=True,namespace_prefix='day1')
            timings.append(dict(job='offline',task=i,protocol=i%6,seconds=time.perf_counter()-t))
            dump_json(bench/'times.json',timings)
        for m in CENTERS:
            t=time.perf_counter();w=World(cfg,'network/day1',70000)
            r=simulate(cfg,w,b,m,adapter if m in ('C','CR_raw') else None,bench/m)
            timings.append(dict(job=m,seconds=time.perf_counter()-t,updates=r['observations_total']))
            dump_json(bench/'times.json',timings)
        return
    if (b['kind'],b['seed'])!=(args.head,args.seed):raise ValueError('Bundle and output head/seed differ')
    adapter=None;batch=jobs(args.block)
    for j,(i,m,mode,frac) in enumerate(batch):
        if j<args.start or (args.stop is not None and j>=args.stop):continue
        path=dest/'network'/f'{args.head}_{args.seed}'/args.block/f'{j:03}';done=path/'completed.json'
        if done.exists():continue
        if m in ('C','CR_raw') and adapter is None:adapter=ChronosAdapter(args.device,args.checkpoint)
        t=time.perf_counter();w=World(cfg,'network/test',20000+i,reusable=frac)
        r=simulate(cfg,w,b,m,adapter,path,action_mode=mode,compact=i!=0)
        row={k:v for k,v in r.items() if np.isscalar(v) or v is None}
        row.update(block=args.block,mission=i,policy=m if mode=='joint' else mode,target_reuse=frac,
                   seconds=time.perf_counter()-t,hold_failure_fraction=r['hold_failures']/r['hold_segments'] if r['hold_segments'] else None)
        temp=done.with_suffix('.tmp');dump_json(temp,row);temp.replace(done)
        print(args.block,j,m,round(row['seconds'],1),'s',flush=True)
if __name__=='__main__':run()

