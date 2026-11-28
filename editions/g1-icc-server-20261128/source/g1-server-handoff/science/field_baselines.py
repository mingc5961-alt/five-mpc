"""External-flight zero-shot evaluation, using only paid samples for prediction."""
import argparse
import zlib
import csv
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from server_paths import RUN
from age_panel import AGES,COUNTS,PRIMARY_A,PRIMARY_N,Geometry,angle_deg
from center import mean_projection
from routeclock.filters import PositionIMM
from routeclock.geometry import angles,position_covariance,spherical_to_xyz,unit
from routeclock.world import Observation
from raft_rebuild import Raft

COLUMNS=['flight','seq','timestamp','position_x','position_y','position_z']
EXTRA_N=[80,160,320]
OUT=RUN/'field'
STD=np.array([1.,*np.deg2rad([.1,.1]),.2])


def prepare():
    df=pd.read_csv(RUN/'inputs/grigoriou.csv',sep=';',usecols=COLUMNS)
    if len(df.dropna())!=len(df) or not np.isfinite(df.to_numpy()).all():
        raise ValueError('non-finite source observations')
    # Fixed virtual station 300 m west of the first reported fix, at its height.
    origin=df.iloc[0][['position_x','position_y','position_z']].to_numpy()
    df['day']=pd.to_datetime(df.timestamp,unit='s',utc=True).dt.strftime('%Y-%m-%d')
    manifest=[];segments=[]
    for (flight,day),group in df.groupby(['flight','day'],sort=True):
        ordered=group.sort_values(['timestamp','seq'],kind='stable').drop_duplicates('timestamp',keep='first')
        t=ordered.timestamp.to_numpy(dtype=float)
        lon,lat,alt=[ordered[k].to_numpy() for k in ('position_x','position_y','position_z')]
        xyz=np.stack([300+6371000*np.deg2rad(lon-origin[0])*np.cos(np.deg2rad(origin[1])),
                      6371000*np.deg2rad(lat-origin[1]),alt-origin[2]],axis=-1)
        dt=np.diff(t); jump=np.linalg.norm(np.diff(xyz,axis=0),axis=1)/np.maximum(dt,1)
        cut=np.flatnonzero((dt>2)|(jump>25))+1
        for i,ids in enumerate(np.split(np.arange(len(t)),cut)):
            tt=t[ids];xx=xyz[ids]
            admitted=len(ids)>=12 and tt[-1]-tt[0]>=11
            manifest.append(dict(flight=int(flight),day=day,part=i,raw_rows_in_flight_day=len(group),samples=len(tt),
                                 start_s=tt[0],end_s=tt[-1],accepted=bool(admitted)))
            if admitted:segments.append((int(flight),day,i,tt,xx))
    return origin,manifest,segments


def prepare_bosello():
    inputs=RUN/'inputs/bosello'
    src=json.loads((inputs/'source.json').read_text())
    manifest=[];segments=[];origin=None
    for i,entry in enumerate(sorted(src['files'],key=lambda a:a['local']),1):
        data=pd.read_csv(inputs/entry['local'])
        flight=entry['local'].split('_')[0]
        cluster=entry['group']+'_'+flight.split('-')[-1]
        t=data.elapsed_time.to_numpy()
        xyz=data[['drone_x','drone_y','drone_z']].to_numpy()
        valid=np.isfinite(t)&np.isfinite(xyz).all(axis=1)
        t=t[valid];xyz=xyz[valid]
        if not len(t):continue
        if origin is None:origin=xyz[0].copy()
        xyz=xyz-origin+np.array([300.,0.,0.])
        dt=np.diff(t)
        speed=np.linalg.norm(np.diff(xyz,axis=0),axis=1)/np.maximum(dt,1e-9)
        cut=np.flatnonzero((dt<=0)|(dt>.25)|(speed>25))+1
        for part,indices in enumerate(np.split(np.arange(len(t)),cut)):
            time=t[indices];pos=xyz[indices]
            ok=len(time)>1 and time[-1]-time[0]>=11
            manifest.append(dict(flight=flight,day=cluster,source_file=entry['member'],part=part,
                                 start_s=float(time[0]),end_s=float(time[-1]),samples=len(time),accepted=bool(ok)))
            if ok:segments.append((flight,cluster,part,time,pos))
    return origin,manifest,segments


def run(source):
    torch.set_num_threads(1)
    dest=OUT if source=='grigoriou' else RUN/'field_bosello'
    dest.mkdir(parents=True,exist_ok=True)
    if source=='grigoriou':
        origin,manifest,segments=prepare()
        meta=dict(source_record='13682870',md5='ce031b8e54719c9f44f3d896a297bd8a',
                  coordinate_origin_lon_lat_z=origin.tolist(),source_timestamp_precision_s=1,
                  grouping='flight label x UTC day; inference clustered by UTC day')
    else:
        origin,manifest,segments=prepare_bosello()
        meta=dict(source_release='https://github.com/tii-racing/drone-racing-dataset/releases/tag/v1.0.0',
                  coordinate_origin_xyz_m=origin.tolist(),max_interpolation_gap_s=.25,
                  grouping='flight identity preserved; exploratory groups by piloted/autonomous x course shape')
    (dest/'manifest.json').write_text(json.dumps(dict(**meta,station_offset_enu_m=[-300,0,0],
        field_tuning=False,segments=manifest),indent=2))
    net=Raft()
    records=[];prefixes=[]
    for flight,day,part,t,xyz in segments:
        t=t-t[0]
        v=np.gradient(xyz,t,axis=0)
        geometry=Geometry(t,xyz,v)
        seed=(int(flight),int(day.replace('-',''))) if source=='grigoriou' else (zlib.crc32(flight.encode()),zlib.crc32(day.encode()))
        rng=np.random.default_rng([271127,*seed,part])
        times=np.cumsum(np.r_[.0031,rng.uniform(3.,4.2,int(t[-1]/3)+2)])
        times=times[times<=t[-1]]
        imm=PositionIMM();hist=[]
        interp=lambda arr,s:np.array([np.interp(s,t,col) for col in arr.T])
        for n,tau in enumerate(times,1):
            p=interp(xyz,tau);vel=interp(v,tau)
            z=np.r_[np.linalg.norm(p),angles(p),unit(p)@vel]+STD*rng.standard_normal(4)
            o=Observation(tau,spherical_to_xyz(z),position_covariance(z,STD),float(z[3]),STD[3],z)
            hist.append(o);imm.observe(o)
            if n not in (*COUNTS,*EXTRA_N):continue
            if tau+min(AGES)>t[-1]:continue
            preds=mean_projection()@(imm.weights[:,None]*imm.x).ravel()
            prefixes.append(list(hist))
            anchors=[]
            for age in AGES:
                if tau+age>t[-1]:continue
                label=geometry.label(tau,tau+age)
                y=interp(xyz,tau+age)
                anchors.append((flight,day,part,n,tau,age,label,int(n==len(times) or times[n]>tau+age),
                                angle_deg(preds[int(2*age)],y),y))
            records.append(anchors)
    dst=dest/'geometry.csv.gz'
    temp=dst.with_suffix('.tmp')
    with gzip.open(temp,'wt',newline='') as f:
        wr=csv.writer(f);wr.writerow(['flight','day','part','n','tau','age','label','natural','e_IMM','e_RAFT'])
        for j in range(0,len(prefixes),48):
            pred=net.predict(prefixes[j:j+48])
            for anchors,c in zip(records[j:j+48],pred):
                for *info,y in anchors:
                    wr.writerow([*info,angle_deg(c[int(2*info[5])],y)])
    temp.replace(dst)
    data=pd.read_csv(dst)
    summary=dict(source_flight_ids=int(data.flight.nunique()),source_clusters=int(data.day.nunique()),
                 accepted_segments=len(segments),all_segments=len(manifest),prefixes=len(prefixes),queries=len(data),
                 revisiting_flights=int(data.loc[data.label=='same_direction_revisit','flight'].nunique()))
    field=data[data.n.isin(PRIMARY_N)&data.age.isin(PRIMARY_A)&data.label.eq('same_direction_revisit')].copy()
    summary.update(primary_queries=0,contributing_clusters=0,paired_cluster_ci95=None)
    flight_ids=sorted(set(r['flight'] for r in manifest))
    per_flight=[]
    for flight in flight_ids:
        a=data.loc[data.flight==flight];b=field.loc[field.flight==flight]
        per_flight.append(dict(flight=flight,queries=len(a),revisiting_queries=int(a.label.eq('same_direction_revisit').sum()),
            primary_queries=len(b),primary_delta_IMM_minus_RAFT_deg=float((b.e_IMM-b.e_RAFT).mean()) if len(b) else None,
            RAFT_worse_fraction=float((b.e_RAFT>b.e_IMM).mean()) if len(b) else None))
    pd.DataFrame(per_flight).to_csv(dest/'per_flight.csv',index=False)
    if len(field):
        field['delta']=field.e_IMM-field.e_RAFT
        # Equal n/age cells within acquisition day, then equal days.
        per=field.groupby(['day','n','age']).delta.mean().groupby(level=0).mean()
        boot=np.random.default_rng(271127).choice(per.to_numpy(),(20000,len(per))).mean(1)
        summary.update(primary_queries=len(field),contributing_clusters=len(per),d_IMM_minus_RAFT_deg=float(per.mean()),
                       paired_cluster_ci95=np.quantile(boot,[.025,.975]).tolist() if len(per)>=2 else None)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',nargs='?',choices=['grigoriou','bosello'],default='grigoriou')
    run(parser.parse_args().source)
