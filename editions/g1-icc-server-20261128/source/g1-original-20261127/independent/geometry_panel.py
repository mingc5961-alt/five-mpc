"""Paired open-loop geometry, expiry and admissibility at fixed paid prefixes."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import csv,gzip,json,multiprocessing,os,time
import numpy as np
import pandas as pd
import torch
from history import ROOT,put
from policy_env import initialize
from paired_grid import BASE,load_checked
from paired_models import parts,tube,trajectory
from predictors import Model
from slots import compute_slot
from panel_types import Geometry,angle_deg,Snapshot,features
OUT=BASE/'geometry_panel'
AGES=(.5,1,2,3,5,8,10);COUNTS={1,2,3,4,6,8,12,24,40}
NAMES=('Acal','M','Fstar','IMM','RAFT','Chronos')
@lru_cache
def snapshots():
    nets=[]
    for age in (True,False):
        net=Snapshot(age);net.load_state_dict(torch.load(BASE/f'auxiliary/models/snapshot_age_{int(age)}.pt',weights_only=True))
        nets.append(net.eval())
    return nets

def life(w,u,e,c,width,factors,radio):
    import paid_full as pf
    from routeclock.radio import array_gain
    ht=pf.network.QUERY_H[1:]
    d=np.arange(min(500,int(np.floor((180-e)*50+1e-9)))+1)*.02
    truth=trajectory(w,u,e+d);pred=np.column_stack([np.interp(d,pf.GRID,x) for x in c.T])
    bad=np.flatnonzero(array_gain(pf.angle(pred),pf.angle(truth),(32,32))/1024<10**(-.3))
    real=d[max(0,bad[0]-1)] if len(bad) else d[-1]
    env=np.maximum(width,np.vstack((width[:1],width[:-1])))
    loss=radio.worst_losses(np.tile(pf.angle(c),(len(ht),1)),(.75*factors[:,None,None]*env).reshape(-1,2),(32,32)).reshape(len(ht),-1)
    ok=np.logical_and.accumulate(np.where(pf.GRID[None,:]<=ht[:,None],loss,0).max(1)<=3)
    nominal=max([0.,*ht[ok]])
    return float(real),float(nominal),int(not len(bad))
def one(job):
    import paid_full as pf
    from grudata import schedule
    from routeclock.filters import PositionIMM
    from routeclock.geometry import angular_jacobian
    split,k=job;folder=OUT/split
    for s in ('scores','receipts','active'):(folder/s).mkdir(parents=True,exist_ok=True)
    dest=folder/'scores'/f'{k}.csv.gz';receipt=folder/'receipts'/f'{k}.json'
    if receipt.exists() and dest.exists():return json.loads(receipt.read_bytes())
    with compute_slot():
        mark=folder/'active'/f'{k}.json';put(mark,dict(pid=os.getpid(),started=time.time()))
        try:
            w,wm=load_checked(split,k);radio=pf.Radio(w.cfg)
            geom=[Geometry(w.times,w.positions[u],w.velocities[u]) for u in range(16)]
            entries=list(schedule(w,41000+k));by={u:[t for uid,t in entries if uid==u] for u in range(16)}
            net=pf.model();states=[pf.State(w.cfg,net,'G') for _ in range(16)]
            imm=Model(w,'IMM');raft=Model(w,'RAFT');chrono=Model(w,'Chronos')
            hs=[[] for _ in range(16)];nrows=0;prefixes=0
            tmp=dest.with_suffix(f'.{os.getpid()}.tmp')
            with gzip.open(tmp,'wt',newline='') as f:
                write=csv.writer(f)
                rowkeys=['world','u','n','tau','age','h','label','group','natural']
                write.writerow(rowkeys+[name+x for name in NAMES for x in ('_deg','_remaining_true','_remaining_admitted','_censored')]+['snapshot_age_deg','snapshot_no_age_deg'])
                for u,t in entries:
                    o=w.observe(u,t);hs[u].append(o);n=len(hs[u]);net.n=n
                    pack=states[u].update(o,net)['G'];ci,wi=imm.predict(u,o)
                    if n not in COUNTS or t+min(AGES)>180:continue
                    val=[]
                    for kind in ('Acal','M','Fstar'):
                        c,feat=parts(pack,kind);model=tube(kind);val.append((c,pf.raw(model['tree'],feat),model['final']))
                    val.append((ci,wi,imm.factors))
                    raft.h[u]=hs[u][:-1].copy();cr,wr=raft.predict(u,o);val.append((cr,wr,raft.factors))
                    cc,wc=chrono.base.update(u,hs[u]);val.append((cc,wc,chrono.factors))
                    limits=[life(w,u,t,c,width,factors,radio) for c,width,factors in val]
                    prefixes+=1
                    for age in AGES:
                        for h in (0,2):
                            if age+h>10 or t+age+h>180:continue
                            y=trajectory(w,u,t+age+h);idx=round(2*(age+h))
                            label=geom[u].label(t,t+age+h,w.detour_windows[u])
                            if w.groups[u]=='no_reusable_path' and label=='first_pass':label='no_reusable_path'
                            cells=[]
                            for (c,_,_),(real,adm,cens) in zip(val,limits):
                                cells.extend((angle_deg(c[idx],y),max(0.,real-age),max(0.,adm-age),cens))
                            with torch.inference_mode():
                                x=torch.tensor([features(o,age,h)],dtype=torch.float32)
                                cells.extend(float(angle_deg(snap(x).numpy()[0],y)) for snap in snapshots())
                            write.writerow([k,u,n,t,age,h,label,w.groups[u],int(n==len(by[u]) or by[u][n]>t+age),*cells]);nrows+=1
            tmp.replace(dest);r=dict(world=k,physical_sha256=wm['physical_sha256'],prefixes=prefixes,queries=nrows)
            put(receipt,r);return r
        finally:mark.unlink(missing_ok=True)
def main():
    initialize();design=json.loads((BASE/'design/frozen.json').read_bytes());worlds=range(2000,2000+design['core_worlds'])
    from paired_design import check_frozen_inputs
    check_frozen_inputs(design)
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('fork')) as pool:
        rs=[f.result() for f in as_completed([pool.submit(one,('confirmation',k)) for k in worlds])]
    clusters=[];alln=0
    for k in worlds:
        data=pd.read_csv(OUT/'confirmation/scores'/f'{k}.csv.gz');alln+=len(data)
        rows=data[data.n.isin([6,8,12,24,40])&data.age.isin([2,3,5])&(data.h==0)&(data.label=='same_direction_revisit')]
        if len(rows):
            d=rows.groupby(['n','age'])[['M_deg','Acal_deg','Fstar_deg']].mean().mean()
            clusters.append([k,len(rows),d['M_deg']-d['Acal_deg'],d['Fstar_deg']-d['Acal_deg']])
    delta=np.asarray([x[2:] for x in clusters])
    summary=dict(worlds=len(rs),queries=alln,primary_world_count=len(clusters),primary_queries=sum(x[1] for x in clusters),primary_worlds=clusters)
    if len(delta)>=2:
        idx=np.random.default_rng(271127).integers(len(delta),size=(20000,len(delta)))
        summary.update(primary_effects=['motion_minus_adaptive_deg','fixed_minus_adaptive_deg'],means=delta.mean(0).tolist(),
                       ci95=np.quantile(delta[idx].mean(1),[.025,.975],axis=0).tolist())
    from geometry_statistics import summarize
    summarize('confirmation',len(rs))
    put(OUT/'summary.json',summary)
if __name__=='__main__':main()
