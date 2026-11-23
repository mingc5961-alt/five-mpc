"""Short-window CMU replay in the common frame of a preregistered virtual station."""
import argparse
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
from routeclock.config import load_config
from routeclock.world import World, rng_for
from .data import BASE,ASSETS
from .network import run_world
from .tubes import MODELS

def local_enu(llh,origin):
    # WGS84 horizontal curvature at the field; relative MSL height from the log.
    lon,lat,alt=origin;phi=np.deg2rad(lat);e2=6.69437999014e-3
    ell=1-e2*np.sin(phi)**2
    nx=6378137./np.sqrt(ell)
    mx=6378137.*(1-e2)/ell**1.5
    return np.column_stack([300+(nx+alt)*np.cos(phi)*np.deg2rad(llh[:,0]-lon),
                            (mx+alt)*np.deg2rad(llh[:,1]-lat),llh[:,2]-alt-10])

class Flight(World):
    def __init__(self,flight,cfg,t,enu):
        self.cfg=cfg;self.n=1;self.namespace='cmu/short';self.mission=flight
        self.dt=.02;self.times=np.arange(5002)*self.dt
        self.positions=np.array([np.column_stack([np.interp(self.times,t,v) for v in enu.T])])
        self.velocities=np.gradient(self.positions,self.dt,axis=1)
        rng=rng_for(self.namespace,flight)
        self.noise=rng.normal(size=(1,len(self.times),4))
        r=cfg['radio'];phi=np.exp(-self.dt/r['shadow_correlation_s'])
        z=rng.normal(0,r['shadow_std_db'],len(self.times))
        for j in range(1,len(z)):z[j]=phi*z[j-1]+np.sqrt(1-phi**2)*z[j]
        self.shadows=z[None]

def flight_windows(phase,cfg):
    meta=pd.read_csv(ASSETS/'cmu/parameters.csv').sort_values(['date','local_time','flight'])
    meta=meta[meta.route.str.startswith('R')]
    dates=sorted(meta.date.unique())
    development=set(dates[:int(np.ceil(.6*len(dates)))])
    with zipfile.ZipFile(ASSETS/'cmu/flights.zip') as archive:
        def read(i):
            with archive.open(f'flights/{i}.csv') as f:return pd.read_csv(f)
        xyz=['position_x','position_y','position_z']
        origin=read(int(meta.iloc[0].flight))[xyz].iloc[0].to_numpy()
        for row in meta.itertuples(index=False):
            if (row.date in development)!=(phase=='development'):continue
            d=read(row.flight);t=d.time.to_numpy();points=d[xyz].to_numpy()
            if not np.isfinite(points[0]).all() or np.any(np.diff(t)<0):continue
            good=np.flatnonzero(np.isfinite(points).all(1)&np.isfinite(t)&(points[:,2]-points[0,2]>5))
            if not len(good):continue
            for ids in np.split(good,np.flatnonzero((np.diff(good)>1)|(np.diff(t[good])>.25))+1):
                if t[ids[-1]]-t[ids[0]]<100.02:continue
                uniq=np.r_[True,np.diff(t[ids])>0];ids=ids[uniq]
                info=dict(flight=row.flight,route=row.route,date=row.date,window_start_s=t[ids[0]],
                          origin_lon_lat_msl=origin.tolist(),station_offset_enu_m=[-300,0,10],max_gap_s=float(np.diff(t[ids]).max()))
                yield info,Flight(row.flight,cfg,t[ids]-t[ids[0]],local_enu(points[ids],origin))
                break
def main(name,phase,limit,pilot):
    cfg=load_config();cfg['simulation'].update(primary_N=1,warmup_s=30.,service_s=60.,sensing_budget=.05/3)
    with (BASE/('pilot' if pilot else 'weights')/(name+'_tubes.pkl')).open('rb') as f:bundle=pickle.load(f)
    model=MODELS[name]()
    for index,(meta,w) in enumerate(flight_windows(phase,cfg)):
        if limit and index>=limit:break
        path=BASE/'real_network'/('pilot' if pilot else 'frozen')/phase/name/str(w.mission)
        path.mkdir(parents=True,exist_ok=True)
        for tube in ('native','lightgbm'):
            dest=path/(tube+'.json')
            if dest.exists():continue
            res,_=run_world(cfg,w,model,bundle,tube)
            tmp=dest.with_suffix('.part');tmp.write_text(json.dumps(dict(**res,**meta,model=name,tube=tube,phase=phase,simulated_radio=True),allow_nan=False));tmp.replace(dest)
        pair=[json.loads((path/(k+'.json')).read_text()) for k in ['native','lightgbm']]
        assert pair[0]['initial_center_sha256']==pair[1]['initial_center_sha256']
        print(name,phase,w.mission,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);p.add_argument('--phase',choices=['development','holdout'],default='development')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    if a.pilot and a.phase!='development':p.error('pilot requires development flights')
    main(a.model,a.phase,a.limit,a.pilot)
