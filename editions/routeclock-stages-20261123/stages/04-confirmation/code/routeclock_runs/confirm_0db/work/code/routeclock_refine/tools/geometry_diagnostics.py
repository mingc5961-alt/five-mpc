import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from routeclock.config import load_config,dump_json
from routeclock.world import World,GROUPS
from routeclock.predictors import Tracker

def run():
    cfg=load_config(True);cfg['simulation']['warmup_s']=120;cfg['simulation']['service_s']=5
    results=[]
    for shape in ['ellipse','figure8','scan']:
        for group in [GROUPS[0],GROUPS[3],GROUPS[4],GROUPS[5]]:
            world=World(cfg,'p0/geometry',GROUPS.index(group),1,[group],shape=shape);tr=Tracker(cfg);errors=[];supports=[]
            for t in np.arange(0,120.01,1):
                tr.observe(world.observe(0,t))
                if t>=60:
                    p,c,s,_,_=tr.memory.forecast([0,1,3,5,10]);truth=np.array([world.truth(0,t+h)[0] for h in [0,1,3,5,10]])
                    errors.append(np.linalg.norm(p-truth,axis=1));supports.append(s)
            results.append({'shape':shape,'group':group,'mean_error_m':np.mean(errors,axis=0),'mean_future_support':np.mean(supports,axis=0),'fragments':len(tr.memory.fragments),'closed':sum(f.closed for f in tr.memory.fragments),'revisits':tr.memory.revisits})
            print(results[-1],flush=True)
    dump_json('runs/p0_geometry.json',results)
if __name__=='__main__':run()
