import argparse
from pathlib import Path
import numpy as np
from .config import load_config,dump_json

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['spine','generate','train','local','matrix','run','select-fixed','probe','summarize','offline']); parser.add_argument('--out',default='runs/local')
    parser.add_argument('--data',default='data/local');parser.add_argument('--bundle',default='runs/local/bundle.pkl')
    parser.add_argument('--profile',choices=['local','formal','model-smoke'],default='local');parser.add_argument('--chronos',action='store_true');parser.add_argument('--device',default='cpu')
    parser.add_argument('--head',choices=['lightgbm','tabpfn'],default='lightgbm');parser.add_argument('--seed',type=int,default=7);parser.add_argument('--count',type=int)
    parser.add_argument('--point-progress',action='store_true');parser.add_argument('--shape',choices=['ellipse','scan','figure8','polygon'])
    parser.add_argument('--fixed-center',choices=['I','RC'],default='I');parser.add_argument('--uavs',type=int)
    parser.add_argument('--split',choices=['train','validation','calibration','test']);parser.add_argument('--methods',default='I,RC');parser.add_argument('--job',type=int,default=0)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    cfg=load_config(local=args.profile!='formal')
    if args.profile=='model-smoke':cfg['simulation']['warmup_s']=12;cfg['simulation']['service_s']=3
    if args.uavs is not None:cfg['simulation']['primary_N']=args.uavs
    if args.command=='matrix':
        from .experiments import experiment_matrix,single_action_matrix
        jobs=experiment_matrix();dump_json(Path(args.out)/'matrix.json',jobs);dump_json(Path(args.out)/'single_actions.json',single_action_matrix());print(f'{len(jobs)} network jobs + 80 single-UAV diagnostics');return
    if args.command=='probe':
        import runpy
        from .config import ROOT
        runpy.run_path(str(ROOT/'tools/probe_models.py'),run_name='__main__');return
    if args.command=='generate':
        from .dataset import generate_anchors
        from .foundation import ChronosAdapter
        adapter=ChronosAdapter(args.device) if args.chronos else None
        splits=[args.split] if args.split else ['train','validation','calibration','test']
        for split in splits:
            count=args.count or (cfg['data']['single_uav_tasks'][split] if args.profile=='formal' else {'train':24,'validation':12,'calibration':12,'test':12}[split])
            generate_anchors(cfg,split,count,Path(args.data)/f'{split}.pkl',adapter,args.profile!='formal',args.shape,args.point_progress)
        dump_json(Path(args.data)/'configuration.json',cfg);return
    if args.command=='train':
        from .dataset import load_anchors
        from .training import train_bundle
        from .evaluation import offline_report
        import json
        cfg=json.loads((Path(args.data)/'configuration.json').read_text(encoding='utf-8'))
        splits=[load_anchors(Path(args.data)/f'{s}.pkl') for s in ['train','validation','calibration','test']]
        bundle,test=train_bundle(*splits,cfg,args.out,args.head,args.device,args.seed);offline_report(test,bundle,args.out);return
    if args.command in ['local','select-fixed','offline','run']:
        from .training import load_bundle
        from .world import World
        from .simulator import simulate
        from .foundation import ChronosAdapter
        if args.command=='run':
            from .experiments import experiment_matrix,run_job
            job=experiment_matrix()[args.job];adapter=ChronosAdapter(args.device) if job['method'] in ['C','CR'] else None
            print(run_job(cfg,job,Path(args.bundle),args.out,adapter));return
        bundle=load_bundle(args.bundle)
        if args.command=='select-fixed':
            from .experiments import select_fixed
            print(select_fixed(cfg,bundle,args.count or 1,Path(args.bundle).parent,args.fixed_center));return
        if args.command=='offline':
            from .dataset import load_anchors
            from .evaluation import offline_report
            offline_report(load_anchors(Path(args.bundle).parent/'test_predictions.pkl'),bundle,args.out);return
        methods=args.methods.split(',');adapter=ChronosAdapter(args.device) if any(m in ['C','CR'] for m in methods) else None
        results=[]
        for mission in range(args.count or 1):
            world=World(cfg,'network/development',30000+mission)
            for method in methods:
                result=simulate(cfg,world,bundle,method,adapter,Path(args.out)/f'{method}_{mission}')
                results.append(result);print({k:result[k] for k in ['method','net_mbps','sensing_fraction','unavailable_s','mean_update_cpu_s']},flush=True)
        dump_json(Path(args.out)/'local_results.json',results);return
    if args.command=='summarize':
        from .evaluation import summarize_network
        print(summarize_network(args.out));return
    from .world import World
    from .radio import Radio
    from .geometry import angles
    cfg=load_config(local=True); cfg['simulation']['warmup_s']=0; cfg['simulation']['service_s']=3
    world=World(cfg,'development/spine',0,n=1); radio=Radio(cfg)
    obs=world.observe(0,0); bits=0; sensed=0; last=0; previous=obs.position.copy(); velocity=np.zeros(3)
    for t in np.arange(0,3,.01):
        if t-last>=.5 or t==0:
            ob=world.observe(0,t)
            if t>0: velocity=(ob.position-previous)/(t-last)
            obs=ob; previous=ob.position.copy(); last=t; sensed+=.0031
            duration=.0069
        else: duration=.01
        pred=obs.position+velocity*(t-obs.t); actual,_=world.truth(0,t)
        expected=radio.snr(pred,angles(pred),(16,16)); snr=radio.snr(actual,angles(pred),(16,16),world.shadow(0,t))
        rate,_=radio.goodput(snr,radio.mcs(expected)); bits+=duration*rate
    result={'real_physics':True,'method':'CV_fixed_development_spine','seconds':3,'sensing_s':sensed,'net_mbps':bits/3/1e6}
    dump_json(Path(args.out)/'spine.json',result); print(result)

if __name__=='__main__': main()
