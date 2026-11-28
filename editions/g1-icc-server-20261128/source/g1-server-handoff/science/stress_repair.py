"""Locally enforce the upstream jerk cap and verify all paired views before scoring."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import pickle
import numpy as np
from stress_views import OUT,CAP,maxima,fingerprint,solve

STRICT=OUT.parent/'stress_strict'
VIEWS=('reference9','detour9','structured0','structured16')


def correct(w):
    records=[]
    for u in range(w.n):
        jerk=np.linalg.norm(np.diff(w.accelerations[u],axis=0),axis=1)/w.dt
        bad=np.flatnonzero(jerk>CAP[2]+1e-6)+1
        if not len(bad):continue
        windows=np.split(bad,np.flatnonzero(np.diff(bad)>round(3/w.dt))+1)
        for events in windows:
            for half_width in (1.,2.,4.):
                start=max(1,events[0]-round(half_width/w.dt))
                end=min(len(w.times)-1,events[-1]+round(half_width/w.dt)+1)
                sl=slice(start,end)
                v=w.velocities[u,sl];a=w.accelerations[u,sl]
                fit=solve(None,None,v,a,w.dt,CAP)
                if fit is None:continue
                pos,dv,da=fit
                pad=np.vstack([w.accelerations[u,start-1],a+da,w.accelerations[u,end]])
                if np.linalg.norm(np.diff(pad,axis=0),axis=1).max()/w.dt>CAP[2]+1e-6:continue
                w.positions[u,sl]+=pos;w.velocities[u,sl]+=dv;w.accelerations[u,sl]+=da
                records.append(dict(user=int(u),begin_s=w.times[start],end_s=w.times[end-1],
                    violation_times_s=w.times[events].tolist(),max_position_change_m=np.linalg.norm(pos,axis=1).max()))
                break
            else:raise RuntimeError(f'no feasible repair, world {w.mission}, user {u}')
    return records


def job(k):
    meta=[]
    ref=None
    for name in VIEWS:
        stem=f'{k}_{name}'
        src=OUT/(stem+'.pkl');dest=STRICT/src.name
        with src.open('rb') as f:w=pickle.load(f)
        raw=json.loads(src.with_suffix('.json').read_text())
        rec=correct(w)
        new=maxima(w.velocities,w.accelerations,w.dt)
        assert np.all(new<=CAP+1e-6)
        pos_res=np.max(abs(np.diff(w.positions,axis=1)-(w.velocities[:,1:]+w.velocities[:,:-1])*w.dt/2))
        vel_res=np.max(abs(np.diff(w.velocities,axis=1)-w.accelerations[:,1:]*w.dt))
        assert pos_res<1e-8 and vel_res<1e-8
        if name=='reference9':ref=w
        if name=='detour9':
            for u in range(w.n):
                mask=np.ones(len(w.times),bool)
                for item in raw['detours']:
                    if item['user']==u:mask&=(w.times<item['onset_s'])|(w.times>item['end_s'])
                for x,y in [(ref.positions,w.positions),(ref.velocities,w.velocities),(ref.accelerations,w.accelerations)]:
                    assert np.array_equal(x[u,mask],y[u,mask])
        assert hashlib.sha256(np.asarray(w.noise).tobytes()).hexdigest()==raw['noise_sha256']
        assert hashlib.sha256(np.asarray(w.shadows).tobytes()).hexdigest()==raw['shadow_sha256']
        if rec:
            tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as f:pickle.dump(w,f,protocol=5)
            tmp.replace(dest)
        elif not dest.exists():dest.symlink_to(os.path.relpath(src,STRICT))
        record=dict(**raw,source_physical_sha256=raw['physical_sha256'],corrections=rec)
        record.update(physical_sha256=fingerprint(w),max_speed_accel_jerk=new.tolist(),
                      position_residual_m=pos_res,velocity_residual_mps=vel_res)
        dest.with_suffix('.json').write_text(json.dumps(record))
        meta.append(record)
    return meta


if __name__=='__main__':
    STRICT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(6) as pool:data=[r for meta in pool.map(job,range(3000,3012)) for r in meta]
    fixes=[c for r in data for c in r['corrections']]
    summary=dict(views=len(data),corrected_views=sum(bool(r['corrections']) for r in data),paired_seeds=12,
       source_scenarios='runs/rebuild_v1/stress',tolerance=1e-6,detours=36,
       max_position_change_m=max(c['max_position_change_m'] for c in fixes),
       full_series_max=np.max([r['max_speed_accel_jerk'] for r in data],axis=0).tolist(),
       position_residual_m=max(r['position_residual_m'] for r in data),velocity_residual_mps=max(r['velocity_residual_mps'] for r in data),
       unchanged_outside_treatment=True,noise_and_shadow_hashes_preserved=True)
    (STRICT/'audit.json').write_text(json.dumps(summary,indent=2))
    print(summary)
