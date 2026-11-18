import pickle
from pathlib import Path
import numpy as np
from .world import World,GROUPS,rng_for
from .predictors import Tracker
from .geometry import angles,angle_error

def observation_schedule(candidates,selected,step,rng):
    end=max(selected);times=[0.]
    while times[-1]<end:times.append(times[-1]+(step if step is not None else rng.uniform(.5,10)))
    return sorted(set([t for t in times if t<=end]+[t for t in candidates if t<=end]))

def generate_anchors(cfg,split,count,out,chronos=None,local=False,shape=None,point_progress=False):
    """One route-template instance per task. Train has 3 anchors, calibration one."""
    records=[]
    formal_groups=[g for g,n in zip(GROUPS,cfg['data']['class_counts'].get(split,[1]*6)) for _ in range(n)]
    for mission in range(count):
        namespace=f'offline/{split}' if shape is None else (f'ood_polygon/{split}' if shape=='polygon' else f'geometry/{shape}/{split}')
        group=GROUPS[mission%6] if local else formal_groups[mission]
        world=World(cfg,namespace,mission,n=1,groups=[group],shape=shape)
        tracker=Tracker(cfg,chronos,point_progress=point_progress); r=rng_for(namespace,mission,777)
        if local:
            end=cfg['simulation']['warmup_s']; anchor_times=[end*.65,end*.82,end]
        else:anchor_times=[r.uniform(a,b) for a,b in cfg['data']['anchor_intervals_s']]
        observation_anchors=list(anchor_times)
        if split=='calibration':anchor_times=[float(r.choice(anchor_times))]
        protocol=mission%6; step=[.5,1,2,5,10,None][protocol]
        times=observation_schedule(observation_anchors,anchor_times,step,r)
        for t in times:
            tracker.observe(world.observe(0,t)); package=tracker.issue(include_chronos=chronos is not None)
            if t not in anchor_times:continue
            truth_h=np.arange(0,10.0001,cfg['simulation']['truth_step_s'])
            truth=np.stack([world.truth(0,t+h)[0] for h in truth_h])
            features={'P':tracker.features(package,'P'),'I':tracker.features(package,'P'),'H':tracker.features(package,'P')}
            if chronos is not None:features.update({'C':tracker.features(package,'P'),'CR_raw':tracker.features(package,'CR_raw')})
            records.append({'task':f'{namespace}/{mission}','group':group,'package':package,'features':features,'truth_h':truth_h,'truth':truth,'shadow':np.array([world.shadow(0,t+h) for h in truth_h]),'sample_h':float(r.choice([.5,1,2,3,5,8,10])),'diagnostics':tracker.memory.diagnostic.copy()})
        print(f'{split} {mission+1}/{count} anchors={len(records)}',flush=True)
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:pickle.dump(records,f)
    return records

def load_anchors(path):
    with Path(path).open('rb') as f:return pickle.load(f)

def center_metrics(anchors,methods):
    rows=[]
    for a in anchors:
        pkg=a['package']
        for method in methods:
            for h in [.5,1,2,3,5,8,10]:
                selected=a['truth_h']<=h+1e-8; hs=a['truth_h'][selected]; truth=a['truth'][selected]
                pred=np.stack([pkg.at(method,float(v)) for v in hs]); error=np.linalg.norm(pred-truth,axis=1)
                ang=np.rad2deg(angle_error(angles(pred),angles(truth)))
                rows.append({'task':a['task'],'group':a['group'],'origin':pkg.origin,'method':method,'horizon':h,'ADE_m':float(error.mean()),'FDE_m':float(error[-1]),'angular_RMSE_deg':float(np.sqrt(np.mean(ang**2))),'angular_p95_deg':float(np.quantile(np.linalg.norm(ang,axis=1),.95))})
    return rows
