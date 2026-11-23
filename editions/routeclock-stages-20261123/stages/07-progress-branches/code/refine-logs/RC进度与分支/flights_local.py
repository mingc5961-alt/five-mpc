import json,pickle
from concurrent.futures import ProcessPoolExecutor
from run_flights import MeasuredFlight,WORK,warm
from core import ROOT,pack,log
from routeclock.config import load_config
from routeclock.predictors import Tracker,apply_gate
from native_map import native_forecast
from refine.network import interpolate
import numpy as np
from extract_new import GATE
manifest=json.loads((WORK/'flight_windows.json').read_text())
def prep(i):
    win=manifest[i];p=ROOT/'flight_inputs'/f'{i:03}.pkl'
    if p.exists():return
    cfg=load_config();cfg['simulation']['primary_N']=1
    world=MeasuredFlight(cfg,win,i);history=warm(world,360,.0031)[0];tr=Tracker(cfg);rows=[]
    with native_forecast():
        for j,o in enumerate(history):
            tr.observe(o);fc=tr.issue()
            if j not in (179,239,299,359):continue
            apply_gate(fc,GATE,fc.features,'RC','P');h=history[:j+1]
            y=interpolate(world.positions[0],o.t+np.arange(501)*.02,world.dt)
            rows.append((i,j,win['route'],h,y,fc,fc.features,pack(h,tr.memory,tr.imm,fc)))
    assert len(rows)==4
    tmp=p.with_suffix('.tmp');tmp.write_bytes(pickle.dumps(rows));tmp.replace(p)
def cases():
    for i in range(len(manifest)):
        yield from pickle.loads((ROOT/'flight_inputs'/f'{i:03}.pkl').read_bytes())
if __name__=='__main__':
    (ROOT/'flight_inputs').mkdir(exist_ok=True)
    with ProcessPoolExecutor(3) as pool:
        for i,_ in enumerate(pool.map(prep,range(len(manifest))),1):
            if i%10==0 or i==len(manifest):log(f'AMOVFLY 实飞外部核验，{i}/{len(manifest)} 架次核对源文件摘要并完成共同历史状态重建，每架次 4 个发布起点。')
