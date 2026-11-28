"""Generate paired composition and temporary deviation views with unchanged noise draws."""
import copy
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from server_paths import RUN
from detour_solver import solve
from worlds import COUNTS, fingerprint, load_world
from routeclock.config import load_config
from routeclock.world import GROUPS, World

OUT = RUN / 'stress'
NAMESPACE = 'g1_icc_server_v1/stress_family'
COMPOSITION = {'reference9': COUNTS, 'structured0': (0, 0, 0, 0, 0, 16),
               'structured16': (5, 2, 2, 2, 5, 0)}
CAP = np.array([12., 4., 8.])


def maxima(v, a, dt):
    return np.array([np.linalg.norm(x, axis=-1).max() for x in
                     (v, a, np.diff(a, axis=-2) / dt)])


def perturb(w, peak):
    """Use a feasible, seeded onset/orientation from the registered 75–105 s interval."""
    view = copy.deepcopy(w)
    metadata = []
    dt = w.dt
    m = round(12 / dt)
    s = np.arange(m + 1) / m
    # Integrate the bump velocity by the same trapezoidal rule as the simulator.
    dv = 1024 * s**3 * (1-s)**3 * (1-2*s) / 12
    dp = np.r_[0., np.cumsum(.5 * (dv[1:]+dv[:-1]) * dt)]
    scale = peak / dp.max()
    dp *= scale
    dv *= scale
    dp[-1] = 0.
    da = np.r_[0., np.diff(dv)/dt]
    for u, group in enumerate(w.groups):
        if group != GROUPS[0]:
            continue
        rng = np.random.default_rng([271127, w.mission, u])
        candidates = [(i, sign) for i in range(75, 106) for sign in (-1, 1)]
        chosen = None
        for j in rng.permutation(len(candidates)):
            onset, sign = candidates[j]
            i = round(onset / dt)
            d = np.array([-w.velocities[u,i,1], w.velocities[u,i,0], 0.])
            if np.linalg.norm(d) < 1e-8:
                continue
            d *= sign / np.linalg.norm(d)
            sl = slice(i, i+m+1)
            v = w.velocities[u,sl] + dv[:,None]*d
            a = w.accelerations[u,sl] + da[:,None]*d
            padded = np.vstack([w.accelerations[u,i-1], a, w.accelerations[u,i+m+1]])
            limits = np.array([np.linalg.norm(v,axis=1).max(), np.linalg.norm(a,axis=1).max(),
                               np.linalg.norm(np.diff(padded,axis=0)/dt,axis=1).max()])
            if (limits <= CAP + 1e-8).all():
                chosen = (onset,sl,d,v,a,limits)
                break
        correction = None
        if chosen is None:
            for j in rng.permutation(len(candidates))[:12]:
                onset, sign = candidates[j]
                i = round(onset/dt)
                sl = slice(i,i+m+1)
                d = np.r_[-w.velocities[u,i,1],w.velocities[u,i,0],0.]
                norm = np.linalg.norm(d)
                if norm < 1e-8:
                    continue
                d *= sign/norm
                candidate = solve(dp,d,w.velocities[u,sl],w.accelerations[u,sl],dt,CAP)
                if candidate is None:
                    continue
                offset,dv_c,da_c = candidate
                v = w.velocities[u,sl]+dv_c
                a = w.accelerations[u,sl]+da_c
                padded = np.vstack([w.accelerations[u,i-1],a,w.accelerations[u,i+m+1]])
                limits = np.r_[np.linalg.norm(v,axis=1).max(),np.linalg.norm(a,axis=1).max(),
                               np.linalg.norm(np.diff(padded,axis=0)/dt,axis=1).max()]
                if (limits <= CAP+1e-6).all() and abs((offset@d).max()-peak)<1e-6:
                    chosen = (onset,sl,d,v,a,limits)
                    correction = offset
                    break
        if chosen is None:
            return None, metadata
        onset,sl,d,v,a,limits = chosen
        offset = dp[:,None]*d if correction is None else correction
        view.positions[u,sl] += offset
        view.velocities[u,sl] = v
        view.accelerations[u,sl] = a
        view.detour_windows[u].append((onset,onset+12))
        metadata.append(dict(user=u,onset_s=onset,end_s=onset+12,
                             lateral_peak_m=float(dp.max()),direction=d.tolist(),window_max=limits.tolist(),
                             norm_peak_m=float(np.linalg.norm(offset,axis=1).max()),
                             constrained_correction=correction is not None))
    assert len(metadata)==3
    return view,metadata


def dev(k):
    k,peak = k
    w = load_world('development',k)
    return perturb(w,peak)[0] is not None


def save(w,name,details):
    dest = OUT/f'{w.mission}_{name}.pkl'
    temp = dest.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        pickle.dump(w,f,protocol=5)
    temp.replace(dest)
    residual = np.diff(w.positions,axis=1)-.5*(w.velocities[:,1:]+w.velocities[:,:-1])*w.dt
    dv = np.diff(w.velocities,axis=1)-w.accelerations[:,1:]*w.dt
    result = dict(view=name,seed=w.mission,namespace=w.namespace,groups=w.groups,dt=w.dt,
        max_speed_accel_jerk=maxima(w.velocities,w.accelerations,w.dt).tolist(),
        position_residual_m=float(np.max(abs(residual))),velocity_residual_mps=float(np.max(abs(dv))),
        physical_sha256=fingerprint(w),noise_sha256=hashlib.sha256(np.asarray(w.noise).tobytes()).hexdigest(),
        shadow_sha256=hashlib.sha256(np.asarray(w.shadows).tobytes()).hexdigest(),detours=details)
    assert result['position_residual_m']<1e-8 and result['velocity_residual_mps']<1e-8
    dest.with_suffix('.json').write_text(json.dumps(result))
    return result


def make(job):
    k,peak = job
    cfg = load_config()
    cfg['simulation'].update(warmup_s=0, service_s=180, sensing_budget=3/180)
    perm = np.random.default_rng(410827+k).permutation(16)
    ref = None
    rows = []
    for name,counts in COMPOSITION.items():
        groups = np.repeat(GROUPS,counts)[perm].tolist()
        w = World(cfg,NAMESPACE,k,groups=groups)
        if name=='reference9':
            ref = w
            d,details = perturb(ref,peak)
            if d is None:
                raise ValueError(f'No feasible frozen detour for seed {k}')
            rows.append(save(d,'detour9',details))
        else:
            w.noise = ref.noise
            w.shadows = ref.shadows
        rows.append(save(w,name,[]))
    assert len({r['noise_sha256'] for r in rows})==len({r['shadow_sha256'] for r in rows})==1
    return rows


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(8) as pool:
        checked = {}
        for peak in (5.,4.,3.,2.,1.):
            checked[peak] = list(pool.map(dev,[(k,peak) for k in range(1400,1408)]))
            if all(checked[peak]):
                break
        else:
            raise ValueError('No common feasible development amplitude')
        (OUT/'frozen_design.json').write_text(json.dumps(dict(development=list(range(1400,1408)),
            candidate_order='seeded shuffle of all integer seconds 75..105 and lateral signs, first physically feasible; then minimum-change constrained kinematic fit at up to 12 seeded windows',
            checks=checked,eligible=GROUPS[0],peak_m=peak,duration_s=12,window_caps=CAP.tolist()),indent=2))
        rows = [r for group in pool.map(make,[(k,peak) for k in range(3000,3012)]) for r in group]
    (OUT/'audit.json').write_text(json.dumps(dict(seeds=12,views=len(rows),peak_m=peak,
         max_over_full_worlds=np.max([r['max_speed_accel_jerk'] for r in rows],axis=0).tolist(),
         max_in_detour_windows=np.max([d['window_max'] for r in rows for d in r['detours']],axis=0).tolist(),
         above_cap_worlds=[(r['seed'],r['view'],r['max_speed_accel_jerk']) for r in rows
                           if (np.array(r['max_speed_accel_jerk'])>CAP+1e-6).any()]),indent=2))
