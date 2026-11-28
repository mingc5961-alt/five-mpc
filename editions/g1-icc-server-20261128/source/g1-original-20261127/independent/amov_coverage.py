"""Inventory cold-start paid-query geometry for every flight in AMOVFLY source metadata."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from history import ROOT,put
sys.path.insert(0,str(ROOT.parent/'g1-server-handoff/science'))
from age_panel import Geometry,AGES,COUNTS,PRIMARY_A,PRIMARY_N

DATA=Path('/data/chen/routeclock_runs/confirm_0db/work/amovfly')
OUT=ROOT/'recovery_01/amov_geometry'
COLUMNS=['time','gps_x','gps_y','gps_z','real_lat','real_long']

def coverage(task):
    row,prior=task
    name=row['FlightName'];group=row['Data Dir']
    path=DATA/group/group/(name+'.csv')
    mark=OUT/'active'/(name+'.json')
    put(mark,dict(pid=os.getpid(),started=time.time()))
    try:
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if prior and sha!=prior:raise ValueError(f'source SHA changed for {name}')
        cols=pd.read_csv(path,usecols=COLUMNS)[COLUMNS].to_numpy(dtype=float)
        t=cols[:,0];xyz=cols[:,1:4];lat,lon=cols[:,4:].T
        valid=np.isfinite(cols).all(1)&(np.abs(lat)<=90)&(np.abs(lon)<=180)&(lat!=0)&(lon!=0)
        dt=np.diff(t);length=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
        linked=valid[1:]&valid[:-1]&(dt>0)&(dt<=.25)&(length<=25*dt)
        bounds=np.r_[0,np.flatnonzero(~linked)+1,len(t)]
        seed=zlib.crc32(name.encode())
        counts=Counter(); segments=0; prim=0;primary_revisits=0
        dest=OUT/'queries'/(name+'.csv.gz')
        temp=dest.with_suffix(f'.{os.getpid()}.tmp')
        with gzip.open(temp,'wt',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(['segment_start_row','paid_prefix_n','last_paid_s','age_s','query_s','geometry','until_next_fix','primary'])
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                if not valid[lo] or hi-lo<2 or t[hi-1]-t[lo]<11:continue
                tt=t[lo:hi]-t[lo];p=xyz[lo:hi]
                geom=Geometry(tt,p,np.gradient(p,tt,axis=0))
                segments+=1
                rng=np.random.default_rng([271127,seed,int(lo)])
                paid=np.cumsum(np.r_[.0031,rng.uniform(3.,4.2,int(tt[-1]/3)+2)])
                paid=paid[paid<=tt[-1]]
                for n in (*COUNTS,80,160,320):
                    if n>len(paid):break
                    tau=paid[n-1]
                    for age in AGES:
                        query=tau+age
                        if query>tt[-1]:break
                        label=geom.label(tau,query);counts[label]+=1
                        primary=n in PRIMARY_N and age in PRIMARY_A
                        prim+=primary
                        primary_revisits+=primary and label=='same_direction_revisit'
                        writer.writerow([lo,n,tau,age,query,label,int(n==len(paid) or paid[n]>query),int(primary)])
        temp.replace(dest)
        result=dict(flight=name,route=row['Route'],session=row['Date time'],group=group,
                    source_sha256=sha,source_rows=len(t),previously_selected=bool(prior),
                    eligible_segments=segments,queries=sum(counts.values()),primary_queries=prim,
                    primary_revisit_queries=primary_revisits,geometry=dict(counts))
        put(OUT/'receipts'/(name+'.json'),result)
        return result
    finally:mark.unlink(missing_ok=True)

def main():
    for sub in ('active','queries','receipts'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    with (DATA/'Flight_info.csv').open(encoding='utf-8-sig') as file:flights=list(csv.DictReader(file))
    assert len(flights)==len({r['FlightName'] for r in flights})
    prior={x['flight_id']:x['source_sha256'] for x in json.loads((DATA.parent/'flight_windows.json').read_bytes())}
    assert len(prior)==59 and set(prior)<=set(r['FlightName'] for r in flights)
    manifest=dict(source_revision=(DATA.parent/'amovfly.revision').read_text().strip(),all_metadata_flights=len(flights),
      prior_selection=len(prior),input='GPS local x,y,z in m; finite valid global fix',segments='split at invalid fixes, time delta <=0 or >0.25s, or speed >25m/s; minimum 11s; history resets',
      schedule='seed [271127, CRC32(flight ID), segment start row]; first fix 0.0031s then U(3,4.2)s',primary_n=sorted(PRIMARY_N),
      primary_ages_s=sorted(PRIMARY_A),geometry_code_sha256=hashlib.sha256((ROOT.parent/'g1-server-handoff/science/age_panel.py').read_bytes()).hexdigest(),
      purpose='external coverage only; no model tuning or scoring',formal_endpoints=0)
    put(OUT/'protocol.json',manifest)
    start=time.time()
    put(OUT/'batch.json',dict(pid=os.getpid(),start=start,total=len(flights),workers=31,phase='coverage'))
    with ProcessPoolExecutor(31,mp_context=multiprocessing.get_context('fork')) as pool:
        jobs=[pool.submit(coverage,(r,prior.get(r['FlightName']))) for r in flights]
        results=[j.result() for j in as_completed(jobs)]
    summary=dict(flights=len(results),with_queries=sum(r['queries']>0 for r in results),
        with_primary_revisits=sum(r['primary_revisit_queries']>0 for r in results),
        primary_revisits=sum(r['primary_revisit_queries'] for r in results),queries=sum(r['queries'] for r in results),
        previous_selection=sum(r['previously_selected'] for r in results),wall_s=time.time()-start)
    put(OUT/'summary.json',summary)
    put(OUT/'batch.json',dict(pid=os.getpid(),start=start,total=len(flights),workers=31,phase='done'))
    print(summary,flush=True)
if __name__=='__main__':main()
