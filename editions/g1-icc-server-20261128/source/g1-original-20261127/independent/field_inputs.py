"""Build identical paid observation prefixes and query keys for all external predictors."""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import csv,gzip,json,os,pickle,sys,time,zlib
from pathlib import Path
import numpy as np,pandas as pd
from history import ROOT,put
from slots import compute_slot
SCI=ROOT.parent/'g1-server-handoff/science';sys.path.insert(0,str(SCI))
from field_baselines import prepare,prepare_bosello,STD
from age_panel import PRIMARY_N,PRIMARY_A
from routeclock.geometry import angles,unit,position_covariance,spherical_to_xyz
from routeclock.world import Observation
BASE=ROOT/'paired_linux_v2/field';PREV=ROOT.parent/'g1-server-handoff/runs/rebuild_v1'
AMOV=Path('/data/chen/routeclock_runs/confirm_0db/work/amovfly')

def queries(path):
    with gzip.open(path,'rt') as file:return list(csv.DictReader(file))
def build_segment(t,xyz,seed,rows):
    t=t-t[0];vel=np.gradient(xyz,t,axis=0)
    interp=lambda arr,s:np.asarray([np.interp(s,t,col) for col in arr.T])
    rng=np.random.default_rng(seed);paid=np.cumsum(np.r_[.0031,rng.uniform(3.,4.2,int(t[-1]/3)+2)])
    paid=paid[paid<=t[-1]];by=defaultdict(list)
    for n,tau,age,label,natural in rows:
        if n>len(paid) or abs(paid[n-1]-tau)>1e-8 or tau+age>t[-1]+1e-8:raise ValueError('paid prefix query mismatch')
        by[n].append((age,label,natural,interp(xyz,tau+age)))
    if not by:return None
    obs=[]
    for tau in paid[:max(by)]:
        p=interp(xyz,tau);v=interp(vel,tau)
        z=np.r_[np.linalg.norm(p),angles(p),unit(p)@v]+STD*rng.standard_normal(4)
        obs.append(Observation(tau,spherical_to_xyz(z),position_covariance(z,STD),float(z[3]),STD[3],z))
    return obs,dict(by)
def write(source,flight,segments):
    path=BASE/'inputs'/f'{source}_{flight}.pkl';tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:pickle.dump(segments,stream,protocol=5)
    tmp.replace(path)
def amov(flight):
    name=flight['FlightName'];group=flight['Data Dir'];raw=AMOV/group/group/(name+'.csv')
    rows=queries(ROOT/'recovery_01/amov_geometry/queries'/f'{name}.csv.gz');by=defaultdict(list)
    for r in rows:by[int(r['segment_start_row'])].append((int(r['paid_prefix_n']),float(r['last_paid_s']),float(r['age_s']),r['geometry'],int(r['until_next_fix'])))
    segments=[]
    with compute_slot():
        if rows:
            cols=pd.read_csv(raw,usecols=['time','gps_x','gps_y','gps_z','real_lat','real_long'])[['time','gps_x','gps_y','gps_z','real_lat','real_long']].to_numpy(dtype=float)
            t=cols[:,0];xyz=cols[:,1:4];lat,lon=cols[:,4:].T
            valid=np.isfinite(cols).all(1)&(np.abs(lat)<=90)&(np.abs(lon)<=180)&(lat!=0)&(lon!=0)
            dt=np.diff(t);diff=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
            linked=valid[1:]&valid[:-1]&(dt>0)&(dt<=.25)&(diff<=25*dt);bounds=np.r_[0,np.flatnonzero(~linked)+1,len(t)]
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                if lo not in by:continue
                d=build_segment(t[lo:hi],xyz[lo:hi]+[300,0,0],[271127,zlib.crc32(name.encode()),int(lo)],by[lo])
                segments.append((flight['Date time'].split(' ')[0],int(lo),*d))
        write('amov',name,segments)
    return len(rows)
def main():
    (BASE/'inputs').mkdir(parents=True,exist_ok=True)
    counts={};ids={}
    for source,loader,folder in [('grigoriou',prepare,'field'),('bosello',prepare_bosello,'field_bosello')]:
        rows=queries(PREV/folder/'geometry.csv.gz');grouped=defaultdict(list)
        for r in rows:grouped[(r['flight'],r['day'],int(r['part']))].append((int(r['n']),float(r['tau']),float(r['age']),r['label'],int(r['natural'])))
        _,manifest,segments=loader();by=defaultdict(list)
        for flight,day,part,t,xyz in segments:
            label=str(flight);rr=grouped[(label,str(day),part)]
            if not rr:continue
            seed=(int(flight),int(day.replace('-',''))) if source=='grigoriou' else (zlib.crc32(label.encode()),zlib.crc32(str(day).encode()))
            by[label].append((str(day),part,*build_segment(t,xyz,[271127,*seed,part],rr)))
        all_names=sorted({str(r['flight']) for r in manifest})
        for name in all_names:write(source,name,by[name])
        ids[source]=all_names;counts[source]=len(rows)
    with (AMOV/'Flight_info.csv').open(encoding='utf-8-sig') as f:all_amov=list(csv.DictReader(f))
    with ProcessPoolExecutor(20) as pool:counts['amov']=sum(pool.map(amov,all_amov))
    ids['amov']=[r['FlightName'] for r in all_amov]
    put(BASE/'inputs.json',dict(flights=ids,queries=counts,primary_n=sorted(PRIMARY_N),primary_ages=sorted(PRIMARY_A),
       paired_protocol='same full flight/segment, noise, paid history and future truth for each model. AMOV includes 59 source flights previously selected. No field tuning.'))
if __name__=='__main__':main()
