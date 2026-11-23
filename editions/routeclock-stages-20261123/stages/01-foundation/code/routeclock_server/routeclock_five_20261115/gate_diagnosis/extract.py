"""Build isolated, causal gate-design rows. Truth is attached only after all inputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
import pickle
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from routeclock.geometry import angles,angle_error
from routeclock.config import load_config
from routeclock.world import World,GROUPS
from routeclock.predictors import Tracker
OUT=Path(__file__).resolve().parent/'results'
BASE=['h_norm','innovation','heading_error','entropy','progress_sd','support','past_route_short','past_route_long','past_imm_short','past_imm_long','past_count_short','past_count_long','disagreement','hold_prob','slow_prob','angle_sensitivity']
EXTRA=['replay_gap_deg','replay_imm_fraction','innovation_change','speed','imm_nis','imm_turn_prob','gap_az_deg','gap_el_deg','uncertainty_ratio']
HORIZONS=(.5,1.,2.,3.,5.,8.,10.)
def err(p,y):return np.linalg.norm(np.rad2deg(angle_error(angles(p),angles(y))),axis=-1)
def causal(tr,pkg,h):
    lag=int(np.ceil(h));j=int(2*h)
    # Observations are 1s apart. Replay compares already-issued forecasts at age lag.
    gaps=[]
    for k in range(1,7):
        obs=tr.history[-k];old=tr.snapshots[-k-lag]
        assert abs(obs.t-old.origin-lag)<1e-7 and obs.t<=pkg.origin
        gaps.append(err(old.at('P',lag),obs.position)-err(old.at('I',lag),obs.position))
    f=pkg.features[j]
    gap=np.rad2deg(angle_error(angles(pkg.at('P',h)),angles(pkg.at('I',h))))
    ratio=np.log((.01+np.trace(pkg.covariance['P'][j]))/(.01+np.trace(pkg.covariance['I'][j])))
    return dict(zip(BASE+EXTRA,[*f[:16],np.mean(gaps),np.mean(np.array(gaps)>0),
              f[1]-np.mean([p.features[0,1] for p in tr.snapshots[-6:-1]]),f[16],f[19],f[21]+f[22],*gap,ratio],strict=True))
def extract(job):
    split,idx=job;path=OUT/'routes'/f'{split}_{idx:02d}.csv'
    if path.is_file():return path.name
    net=split=='network'
    n=3 if net else 1
    ids=[3*idx+k for k in range(n)] if net else [idx]
    namespace=('long-history-deadline-v1/'+('train' if split=='validation' else split)) if split in ('train','validation','network') else 'gate-diagnosis-v1/'+split
    cfg=load_config();cfg['simulation'].update(primary_N=3,warmup_s=180,service_s=18)
    w=World(cfg,namespace,idx,n=n,groups=[GROUPS[i%6] for i in ids])
    source=None
    if split in ('train','validation','network'):
        with (ROOT/'runs/five_pilot'/f'{split}_{idx:02}.pkl').open('rb') as fh:source=pickle.load(fh)[0]
    rows=[]
    for uid in range(n):
        tracker=Tracker(cfg);route_hash=hashlib.sha256(w.references[uid].tobytes()).hexdigest()
        for step in range(180):
            t=step+.5+.1*uid; tracker.observe(w.observe(uid,t))
            # Earliest scored anchor 119; completed-error archive is 60s plus 10s forecast.
            if step<46:continue
            pkg=tracker.issue()
            if step not in (119,149,179):continue
            if step==179 and source:
                a=source[uid]
                assert a['route_hash']==route_hash
                np.testing.assert_allclose(pkg.features[:,:16],a['features']['P'][:,:16],atol=1e-8)
                for name in ['P','I']:np.testing.assert_allclose(pkg.centers[name],a['package'].centers[name],atol=1e-8)
            for h in HORIZONS:
                row=dict(split=split,index=idx,uid=uid,route=f'{split}/{idx}/{uid}',route_hash=route_hash,group=w.groups[uid],step=step,t=t,h=h)
                row.update(causal(tracker,pkg,h))
                p=pkg.at('P',h);i=pkg.at('I',h);y=w.truth(uid,t+h)[0]
                if step==179 and source:np.testing.assert_allclose(y,source[uid]['truth'][round(h/.02)],atol=1e-8)
                for label,vec in [('p',p),('i',i),('y',y)]:
                    row.update({label+k:x for k,x in zip('xyz',vec)})
                ep=err(p,y);ei=err(i,y)
                row.update(p_deg=ep,i_deg=ei,imm_win=int(ei<ep),gain_deg2=ep**2-ei**2)
                rows.append(row)
    tmp=path.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(path);return path.name
if __name__=='__main__':
    (OUT/'routes').mkdir(parents=True,exist_ok=True)
    jobs=[(s,i) for s,ids in [('train',range(12)),('validation',range(12,18)),('network',range(2)),('extra_train',range(36)),('audit',range(24))] for i in ids]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for j,name in enumerate(pool.map(extract,jobs)):print(j+1,name,flush=True)

