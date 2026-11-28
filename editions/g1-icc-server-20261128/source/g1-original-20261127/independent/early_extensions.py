"""Release eligible deterministic extension coordinates as soon as one family member is infeasible."""
from collections import defaultdict
import json,os,time
from history import put
from policy_env import initialize
from paired_grid import BASE,SCALES
from paired_design import capacity,config,seeds,data,run_groups,LEARNED,base_path
OUT=BASE/'extensions'
def main():
    initialize();OUT.mkdir(parents=True,exist_ok=True)
    families=defaultdict(list)
    for r in capacity():
        if r['world_first']=='1400':families[(r['scenario'],r['row'][0])].append(r['row'])
    ready=[];families_ready=[]
    for (scene,prefix),rows in families.items():
        b,q=config(scene)
        valid=[r for r in rows if r not in LEARNED and all(base_path(scene,k,r,None,b,q,m).is_file() for k in range(1400,1408) for m in SCALES)]
        if any(not data(scene,r,SCALES,OUT)[0]['feasible'] for r in valid):
            families_ready.append((scene,prefix))
            ready.extend(('development',scene,k,r,None,(.30,1.25),str(OUT/scene)) for r in valid for k in range(1400,1408))
    start=time.time();put(OUT/'early_plan.json',dict(families=families_ready,local_extension_endpoints=2*len(ready),confirmation_scores_read=False))
    run_groups(ready)
    put(OUT/'early_summary.json',dict(completed=2*len(ready),elapsed_s=time.time()-start))
if __name__=='__main__':main()
