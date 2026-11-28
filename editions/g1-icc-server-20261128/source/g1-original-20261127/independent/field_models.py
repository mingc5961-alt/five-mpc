"""External zero-shot LOS comparisons of original adaptive, pure motion and frozen fusion models."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv,gzip,json,multiprocessing,os,pickle,time
from pathlib import Path
import numpy as np
from history import put
from policy_env import initialize
from paired_grid import BASE
from paired_models import parts
from predictors import raft_model
from slots import compute_slot
OUT=BASE/'field'
def angle(p,y):return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(p,y)),np.dot(p,y))))
def work(job):
    import paid_full as pf
    source,flight=job;tag=source+'_'+flight;dst=OUT/'receipts'/(tag+'.json')
    if dst.exists():return json.loads(dst.read_bytes())
    with compute_slot():
        mark=OUT/'active'/(tag+'.json');put(mark,dict(pid=os.getpid(),started=time.time()))
        try:
            with (OUT/'inputs'/(tag+'.pkl')).open('rb') as f:segments=pickle.load(f)
            target=OUT/'scores'/(tag+'.csv.gz');tmp=target.with_suffix(f'.{os.getpid()}.tmp')
            scores=[];prefixes=[]
            for day,part,obs,by in segments:
                net=pf.model();state=pf.State(pf.load_config(),net,'G')
                for i,o in enumerate(obs,1):
                    net.n=i;packet=state.update(o,net)['G']
                    if i not in by:continue
                    adaptive=pf.forecast(packet,'G',1)[0];motion=packet[0];static=parts(packet,'Fstar')[0]
                    prefixes.append(obs[:i]);scores.append((day,part,i,by[i],adaptive,motion,static))
            total=0;primary=0
            with gzip.open(tmp,'wt',newline='') as f:
                wr=csv.writer(f);wr.writerow(['cluster','part','n','age','label','natural','adaptive_deg','motion_deg','Fstar_deg','RAFT_deg','Chronos_deg'])
                for j in range(0,len(prefixes),12):
                    hs=prefixes[j:j+12]
                    raft=raft_model().predict(hs)
                    trained=pf.chrono();trained.torch.set_num_threads(1)
                    chronos=trained.batch(hs)
                    for (day,part,n,queries,a,m,s),r,ch in zip(scores[j:j+12],raft,chronos,strict=True):
                        for age,label,natural,y in queries:
                            idx=int(2*age)
                            wr.writerow([day,part,n,age,label,natural,*[angle(c[idx],y) for c in (a,m,s,r,ch[0])]])
                            total+=1;primary+=int(n in (6,8,12,24,40) and age in (2,3,5) and label=='same_direction_revisit')
            tmp.replace(target)
            receipt=dict(source=source,flight=flight,prefixes=len(prefixes),queries=total,primary_revisit_queries=primary)
            put(dst,receipt);return receipt
        finally:mark.unlink(missing_ok=True)
def main():
    initialize()
    design=json.loads((OUT/'inputs.json').read_bytes());jobs=[(s,n) for s,names in design['flights'].items() for n in names]
    for s in ('receipts','active','scores'):(OUT/s).mkdir(parents=True,exist_ok=True)
    start=time.time()
    put(OUT/'batch.json',dict(total=len(jobs),workers=16,start=start,pid=os.getpid(),phase='external'))
    with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('fork')) as pool:
        rows=[f.result() for f in as_completed([pool.submit(work,j) for j in jobs])]
    by={s:dict(flights=sum(x['source']==s for x in rows),queries=sum(x['queries'] for x in rows if x['source']==s),
        evaluable_flights=sum(x['source']==s and x['queries']>0 for x in rows),
        primary_revisits=sum(x['primary_revisit_queries'] for x in rows if x['source']==s)) for s in design['flights']}
    if any(v['queries']!=design['queries'][s] for s,v in by.items()):raise ValueError('missing shared queries')
    put(OUT/'summary.json',dict(sources=by,wall_s=time.time()-start,closed_loop_runs=0))
    put(OUT/'batch.json',dict(total=len(jobs),workers=16,start=start,pid=os.getpid(),phase='done'))
if __name__=='__main__':main()
