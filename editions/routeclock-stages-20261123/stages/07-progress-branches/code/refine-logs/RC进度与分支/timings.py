"""Single-event service latency after warm initialization, on an unloaded experiment queue."""
from time import perf_counter_ns
import numpy as np,pandas as pd,torch
from replay_new import setup,rr,ARMS,Model,head_for
from features import Advanced
from graph_encoder import enable
from refine import network
from core import ROOT,log
def measure():
    setup();torch.set_num_threads(1);w=40;world=rr.get_world(w);rows=[];engines={}
    for arm in ARMS:
        key='Chronos-2' if arm=='Chronos-native' else arm
        gpu=key in ('Chronos-2','VECTOR','RAFT')
        if gpu and key not in engines:
            engines[key]=Advanced(key)
            if key=='Chronos-2':enable(engines[key].net.net)
        with rr.memoized_arc():
            model=Model(arm,world,engines.get(key),head_for(arm)[0]);hs=network.warm_history(world,180,.0031)
            _=model.batch(hs)
            for step in range(10):
                for uid,h in enumerate(hs):
                    o=world.observe(uid,h[-1].t+1.);h.append(o)
                    if gpu:torch.cuda.synchronize()
                    start=perf_counter_ns();model.batch([h])
                    if gpu:torch.cuda.synchronize()
                    elapsed=(perf_counter_ns()-start)/1e6
                    if step>=2:rows.append((arm,w,step,uid,elapsed))
    frame=pd.DataFrame(rows,columns=['arm','world','step','uid','ms']);assert len(frame)==7*128
    frame.to_csv(ROOT/'单次刷新耗时.csv',index=False);log('关闭实验并发后，7 套系统各用同一 128 次单用户刷新测中心及尺度管服务耗时；GPU 同步计时，包含 RC 状态更新，计时排除传感器与调度。')
if __name__=='__main__':
    import time
    while not (ROOT/'flight_oracle.csv').exists():time.sleep(3)
    measure()
