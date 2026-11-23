"""A hindsight bound on what changing only the ISSUED route/IMM mixing weights can recover."""
import csv
import gzip
import pickle
import numpy as np
from compare import OUT,CACHE,NAMESPACE,network,world_cache,load_config,angles,angle_error
from routeclock.predictors import apply_gate

def run():
    cfg=load_config();cfg['simulation']['sensing_budget']=.05/3;world_cache.BASE=CACHE
    params=pickle.loads((CACHE/'classical.pkl').read_bytes())['RC']
    chronos={}
    for f in (OUT.parent).glob('anchors_*.csv'):
        for row in csv.DictReader(f.open()):
            if row['domain']=='合成' and row['model']=='Chronos-2':
                chronos[int(row['unit']),int(row['user']),int(row['h'])]=row
    a=np.linspace(0,1,1001)
    dst=OUT/'门控与专家上限.csv';temp=dst.with_suffix('.part')
    with temp.open('w') as fp:
        wr=csv.writer(fp);wr.writerow(['world','uid','horizon_s','model','position_m','angle_deg','angular_weight','position_weight'])
        for w in range(40):
            world=world_cache.paired_world(cfg,NAMESPACE,w)
            with gzip.open(CACHE/'warm'/f'v1_{w:03}.pkl.gz','rb') as f:trackers=pickle.load(f)
            for uid,tr in enumerate(trackers):
                p=tr.snapshots[-1];apply_gate(p,params['gate'],p.features,'RC','P')
                for h in (1,3,5,10):
                    k=h*2
                    truth=network.interpolate(world.positions[uid],np.array([p.origin+h]),world.dt)[0]
                    def angle(x):return np.rad2deg(np.linalg.norm(angle_error(angles(x),angles(truth)),axis=-1))
                    I,P,R=[p.centers[m][k] for m in ('I','P','RC')]
                    g=p.diagnostics['gate_weights']['RC'][k]
                    pos_weight=np.clip((truth-I)@(P-I)/max(1e-15,(P-I)@(P-I)),0,1)
                    weights=np.r_[a,g,pos_weight];deg=angle(I+(P-I)*weights[:,None])
                    best=deg.argmin()
                    for name,center in [('I',I),('P',P),('RC',R)]:
                        wr.writerow([w,uid,h,name,np.linalg.norm(center-truth),angle(center),g if name=='RC' else '',g if name=='RC' else ''])
                    row=chronos[w,uid,h]
                    wr.writerow([w,uid,h,'tuned',row['position_m'],row['angle_deg'],'',''])
                    wr.writerow([w,uid,h,'oracle',np.linalg.norm(I+(P-I)*pos_weight-truth),deg[best],weights[best],pos_weight])
    temp.replace(dst)
if __name__=='__main__':run()
