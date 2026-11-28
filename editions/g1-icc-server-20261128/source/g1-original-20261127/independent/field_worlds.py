"""Prepare unretimed measured trajectories and matched counterfactual radio inputs."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from history import ROOT, put
from physical_inputs import World, fingerprint, load_config, sha_file

OUT = ROOT / 'paired_linux_v2/field_extension'
AMOV = Path('/data/chen/routeclock_runs/confirm_0db/work/amovfly')
NEURO = OUT / 'sources/neurobem'

def longest(t, xyz, valid, max_gap, speed_limit=None):
    dt = np.diff(t)
    links = valid[:-1] & valid[1:] & (dt > 0) & (dt <= max_gap)
    if speed_limit is not None:
        links &= np.linalg.norm(np.diff(xyz, axis=0), axis=1) <= speed_limit * dt
    edges = np.r_[0, np.flatnonzero(~links) + 1, len(t)]
    segments = [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi-lo >= 2]
    if not segments:
        return None
    lo, hi = max(segments, key=lambda v:t[v[1]-1]-t[v[0]])
    return t[lo:hi], xyz[lo:hi]

def save_track(source, flight, label, path, segment):
    if segment is None:
        return None
    t, p = segment
    dest = OUT / 'tracks' / (source + '_' + flight + '.npz')
    np.savez_compressed(dest, t=t-t[0], p=p)
    return dict(id=flight, source=source, label=label, seconds=float(t[-1]-t[0]),
                source_start=float(t[0]), sample_count=len(t), track=str(dest), track_sha256=sha_file(dest),
                original_file=str(path), original_sha256=sha_file(path))

def amov_tracks():
    with (AMOV/'Flight_info.csv').open(encoding='utf-8-sig') as f: manifest=list(csv.DictReader(f))
    for r in manifest:
        group, flight = r['Data Dir'],r['FlightName']
        path=AMOV/group/group/(flight+'.csv')
        d=pd.read_csv(path,usecols=['time','gps_x','gps_y','gps_z','real_lat','real_long'])
        t=d.time.to_numpy(); xyz=d[['gps_x','gps_y','gps_z']].to_numpy()
        valid=np.isfinite(d).all(axis=1).to_numpy() & d.real_lat.between(-90,90).to_numpy() & d.real_long.between(-180,180).to_numpy() & (d.real_lat!=0).to_numpy() & (d.real_long!=0).to_numpy()
        rec=save_track('amov',flight,'random' if group=='Random' else 'planned',path,longest(t,xyz,valid,.25,25))
        if rec:yield rec|dict(task=r['Route'],date=r['Date time'].split()[0])

def neuro_tracks():
    groups=defaultdict(list)
    for r in json.loads((NEURO/'index.json').read_bytes()):groups[r['flight']].append(r)
    for flight,parts in sorted(groups.items()):
        best=None
        for r in sorted(parts,key=lambda v:v['member']):
            path=NEURO/r['file']
            if sha_file(path)!=r['sha256']:raise ValueError(path)
            d=pd.read_csv(path,usecols=['t','pos x','pos y','pos z'])
            seg=longest(d.t.to_numpy(),d[['pos x','pos y','pos z']].to_numpy(),np.isfinite(d).all(axis=1).to_numpy(),.025)
            if seg is not None and (best is None or np.ptp(seg[0])>np.ptp(best[1][0])):best=(path,seg)
        if best:
            label=parts[0]['label']
            rec=save_track('neurobem',flight,'random' if 'random points' in label.lower() else 'planned',*best)
            yield rec|dict(task=label,date=flight[:10])

def prepare(source):
    (OUT/'tracks').mkdir(parents=True,exist_ok=True)
    rows=list(amov_tracks() if source=='amov' else neuro_tracks())
    rng=np.random.default_rng(271128 if source=='amov' else 271129)
    if source=='amov': duration=180; n_random=7; panels=2
    else:
        # Use the longest common horizon with two disjoint arms of eight; one planned-only matched control.
        duration=next((s for s in (60,45,30,20) if sum(r['label']=='random' and r['seconds']>=s for r in rows)>=8 and
                       sum(r['label']=='planned' and r['seconds']>=s for r in rows)>=16),None)
        if duration is None:raise ValueError('no admissible NeuroBEM duration')
        n_random=8;panels=1
    pools={}
    for kind in ('random','planned'):
        a=sorted((r for r in rows if r['label']==kind and r['seconds']>=duration),key=lambda r:r['id'])
        pools[kind]=[a[i] for i in rng.permutation(len(a))]
    if len(pools['planned'])<16*panels or len(pools['random'])<n_random*panels:raise ValueError('not enough distinct flight IDs')
    designs=[]
    for i in range(panels):
        random=pools['random'][i*n_random:(i+1)*n_random]
        planned=pools['planned'][i*16:(i+1)*16]
        order=rng.permutation(16); noise_seed=27112800+100*(source=='neurobem')+i
        for view,flights in [('mixed',random+planned[n_random:]),('planned',planned)]:
            flights=[flights[j] for j in order]
            designs.append(dict(id=f'{source}_{i}_{view}',cohort=i,source=source,view=view,duration=duration,
                  quota=round(849*duration/180),noise_seed=noise_seed,flights=flights,station_offset_m=[300.,0.,0.]))
    result=dict(source=source,tracks=rows,panels=designs,statistical_unit='flight for open geometry, independent flight assembly for closed-loop; paired views and policy seeds are dependent')
    put(OUT/(source+'_inputs.json'),result)
    return result

def world(design):
    cfg=load_config();dt=cfg['simulation']['truth_step_s'];duration=design['duration']
    cfg['simulation'].update(warmup_s=0,service_s=duration,sensing_budget=3/180,primary_N=16)
    w=World.__new__(World);w.cfg=cfg;w.n=16;w.dt=dt;w.duration=duration
    w.times=np.arange(round(duration/dt)+1)*dt;w.positions=[];w.velocities=[];w.noise=[];w.shadows=[]
    phi=np.exp(-dt/cfg['radio']['shadow_correlation_s']);sd=cfg['radio']['shadow_std_db']
    for uid,r in enumerate(design['flights']):
        file=Path(r['track'])
        if sha_file(file)!=r['track_sha256']:raise ValueError(file)
        with np.load(file) as d:t,p=d['t'],d['p']+design['station_offset_m']
        offset=(t[-1]-duration)/2; query=w.times+offset
        if query[0]<t[0]-1e-9 or query[-1]>t[-1]+1e-9:raise ValueError('trace bounds')
        w.positions.append(np.column_stack([np.interp(query,t,col) for col in p.T]))
        # Only past samples enter the velocity of a simulated measurement.
        old=np.maximum(t[0],query-.10)
        past=np.column_stack([np.interp(old,t,col) for col in p.T])
        w.velocities.append((w.positions[-1]-past)/np.maximum(1e-12,query-old)[:,None])
        rng=np.random.default_rng([design['noise_seed'],uid])
        w.noise.append(rng.normal(size=(len(query),4)))
        white=rng.normal(size=len(query))*sd;sh=white.copy()
        for j in range(1,len(sh)):sh[j]=phi*sh[j-1]+np.sqrt(1-phi**2)*white[j]
        w.shadows.append(sh)
    w.positions=np.array(w.positions);w.velocities=np.array(w.velocities)
    return w,fingerprint(w)[0]

if __name__=='__main__':
    import sys
    for source in sys.argv[1:]:
        result=prepare(source)
        print(source,len(result['tracks']),[(p['id'],p['duration']) for p in result['panels']],flush=True)
