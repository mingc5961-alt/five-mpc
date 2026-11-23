"""Causal GPS/xyz CSV route matching. No radio throughput inferred from GPS alone."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from routeclock.config import load_config,dump_json
from routeclock.world import Observation
from routeclock.predictors import Tracker
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--time',default='time');p.add_argument('--xyz',default='x,y,z');p.add_argument('--gps',action='store_true');p.add_argument('--sigma',type=float,default=1.);p.add_argument('--out',default='runs/real_matching.json');a=p.parse_args()
df=pd.read_csv(a.csv).sort_values(a.time);coords=df[a.xyz.split(',')].to_numpy(float);times=df[a.time].to_numpy(float);times-=times[0]
if a.gps:
    lon,lat,alt=coords.T;R=6371000.;coords=np.c_[R*np.deg2rad(lon-lon[0])*np.cos(np.deg2rad(lat[0])),R*np.deg2rad(lat-lat[0]),alt-alt[0]]
tracker=Tracker(load_config());rows=[]
for k,(t,p) in enumerate(zip(times,coords)):
    if k and t<=times[k-1]:continue
    obs=Observation(float(t),p,np.eye(3)*a.sigma**2,0.,1e6,np.zeros(4));tracker.observe(obs);pkg=tracker.issue()
    rows.append({'t':t,'observed':p,'memory':tracker.memory.diagnostic,'support':pkg.diagnostics['support'],'progress_std':pkg.diagnostics['progress_std'],'forecast':pkg.centers['P']})
dump_json(a.out,rows)
