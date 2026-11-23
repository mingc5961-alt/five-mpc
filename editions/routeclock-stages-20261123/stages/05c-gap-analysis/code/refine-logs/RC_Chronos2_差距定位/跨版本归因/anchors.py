"""Compare input representations and checkpoint on the same causal service histories."""
import csv
import sys
from pathlib import Path
import numpy as np
from compare import OUT,CACHE,WORLD_IDS,NAMESPACE,predictor,world_cache,network,load_config,angles,angle_error
from routeclock.radio import array_gain

def run():
    cfg=load_config();cfg['simulation']['sensing_budget']=.05/3
    world_cache.BASE=CACHE
    cases=[]
    for i in range(40):
        w=world_cache.paired_world(cfg,NAMESPACE,i)
        warm=network.warm_history(w,180,.0031)
        for uid,hist in enumerate(warm):
            # One paid round-robin observation per 2.976 s per user, through second 300.
            spacing=16*.0031/(.05/3)
            ts=np.arange(180+.0031+uid*.0031/(.05/3),300,spacing)
            h=hist+[w.observe(uid,float(t)) for t in ts]
            a=h[-1].t
            truth=network.interpolate(w.positions[uid],a+np.arange(501)*.02,w.dt)
            cases.append((i,uid,h,truth))
    assert len(cases)==640
    out=OUT/'同历史输入与训练.csv';tmp=out.with_suffix('.part')
    with tmp.open('w') as f:
        wr=csv.writer(f);wr.writerow(['world','uid','model','horizon_s','position_m','angle_deg','safe32','history_count'])
        for name in ('old','old360','old360nopast','frozen','tuned'):
            net=predictor(name)
            for start in range(0,len(cases),8):
                batch=cases[start:start+8]; forecasts=net.batch([h for _,_,h,_ in batch])
                for (w,uid,h,truth),(p,_) in zip(batch,forecasts,strict=True):
                    ts=np.arange(501)*.02
                    pred=np.column_stack([np.interp(ts,network.HS,axis) for axis in p.T])
                    loss=10*np.log10(1024/array_gain(angles(pred),angles(truth),(32,32)))
                    for hz in (1,3,5,10):
                        j=50*hz;wr.writerow([w,uid,name,hz,np.linalg.norm(pred[j]-truth[j]),np.rad2deg(np.linalg.norm(angle_error(angles(pred[j]),angles(truth[j])))),int(np.max(loss[:j+1])<=3),len(h)])
            del net
            f.flush()
    tmp.replace(out)
if __name__=='__main__':run()
