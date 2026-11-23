"""Freeze the C3 regular-loop cohort before consulting any radio results."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
LENGTH = 370.02
SIDES = {"FAVS_R1": 3, "FAVS_R2": 4}
# Fixed field reference from the initial measured fix of the first FAVS flight
# in the author's metadata. All admitted flights are within this field.
def origin(meta):
    row = meta[meta.route.isin(SIDES)].iloc[0]
    fix = pd.read_csv(ROOT / row.source, nrows=1).iloc[0]
    return [float(fix.real_lat), float(fix.real_long)]

def horizontal(lat, lon, ref):
    """WGS84 ECEF position difference projected to the site's east/north axes."""
    phi, lam = np.deg2rad([lat, lon])
    north, east = np.deg2rad(ref)
    ecc = 6.6943799901413165e-3
    def xyz(p, l):
        radius = 6378137.0 / np.sqrt(1 - ecc * np.sin(p)**2)
        return np.stack((radius*np.cos(p)*np.cos(l), radius*np.cos(p)*np.sin(l),
                         radius*(1-ecc)*np.sin(p)), axis=-1)
    diff = xyz(phi, lam) - xyz(north, east)
    return diff @ np.array([[-np.sin(east), -np.sin(north)*np.cos(east)],
                            [np.cos(east), -np.sin(north)*np.sin(east)],
                            [0, np.cos(north)]])

def runs(mask):
    return np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)

def select(row, ref):
    raw = (ROOT / row.source).read_bytes()
    d = pd.read_csv(io.BytesIO(raw))
    t = d.time.to_numpy()
    xy = horizontal(d.real_lat, d.real_long, ref)
    aims = horizontal(d.aim_lat,d.aim_long,ref)
    ground = float(np.median(d.gps_z[t < t[0] + 1]))
    # Separate commanded legs. A returning corner must be closer to its
    # previous lap's position than to either adjacent corner.
    ix = np.r_[0, 1 + np.flatnonzero(np.linalg.norm(np.diff(aims, axis=0),axis=1)>5)]
    corners = aims[ix]
    sides = SIDES[row.route]
    periodic = np.zeros(len(t),bool)
    for j in range(sides,len(ix)):
        a = corners[j-sides:j+1]
        if np.linalg.norm(a[0]-a[-1]) < .5*min(np.linalg.norm(a[0]-a[1]),np.linalg.norm(a[-1]-a[-2])) and np.max(np.linalg.norm(a,axis=1)) < 500:
            periodic[ix[j-sides]:ix[j]+1] = True
    # The recorded route occasionally skips a single corner inside an otherwise
    # continuous mission. Keep these within-route legs, and split on an absent
    # target or departure from the verified polygon's bounding box.
    if not periodic.any(): return None
    reference = aims[periodic & (np.linalg.norm(aims,axis=1)<500)]
    first, last = np.flatnonzero(periodic)[[0,-1]]
    periodic[first:last+1] = True
    periodic &= ((aims >= reference.min(0)-5) & (aims <= reference.max(0)+5)).all(axis=1)
    good = periodic & np.isfinite(d[["gps_x","gps_y","gps_z","real_lat","real_long"]]).all(axis=1).to_numpy()
    good &= (np.linalg.norm(xy,axis=1)<500) & (d.gps_z.to_numpy() >= ground+5)
    links = good[1:] & good[:-1] & (np.diff(t)>0) & (np.diff(t)<=.25)
    for lo,last in runs(links):
        if t[last]-t[lo] < LENGTH: continue
        end = lo + np.searchsorted(t[lo:last+1],t[lo]+LENGTH)
        # Measured angular winding around the reference polygon's centroid.
        use = ix[(ix>=lo)&(ix<=end)]
        center = np.median(corners[np.searchsorted(ix,use)][:len(use)//sides*sides].reshape(-1,sides,2),axis=0).mean(axis=0)
        p = xy[lo:end+1]-center
        phase = np.unwrap(np.arctan2(p[:,1],p[:,0]))
        at = np.interp(t[lo]+np.array([0,180,360]),t[lo:end+1],phase)
        return dict(flight_id=row.flight_id,site_id="AMOVFLY_field",route=row.route,session=row.session,
                    uav=row.flight_id.split('_')[0],source=row.source,source_sha256=hashlib.sha256(raw).hexdigest(),
                    first_t=t[lo],warm_end=t[lo]+180,score_end=t[lo]+360,truth_end=t[lo]+LENGTH,
                    gap_max=np.diff(t[lo:end+1]).max(),ground_z=ground,site_ref=ref,
                    station_enu=[-100,0,5],warm_lap_equivalents=float(abs(at[1]-at[0])/(2*np.pi)),
                    score_lap_equivalents=float(abs(at[2]-at[1])/(2*np.pi)))
    return None

def main():
    meta=pd.read_csv(WORK/'amovfly_quality_candidates.csv')
    ref=origin(meta)
    selected=[]; audit=[]
    for r in meta.itertuples():
        reason=r.status
        if reason=='quality_candidate':
            if r.route not in SIDES: reason='outside_regular_loop_cohort'
            else:
                win=select(r,ref)
                if win: selected.append(win)
                reason='included' if win else 'insufficient_continuous_periodic_interval'
        audit.append(dict(flight_id=r.flight_id,route=r.route,reason=reason))
    assert selected and len(audit)==270 and len({r['flight_id'] for r in selected})==len(selected)
    pd.DataFrame(audit).to_csv(WORK/'flight_exclusions.csv',index=False)
    out=WORK/'flight_windows.json'
    payload=json.dumps(selected,indent=2,allow_nan=False)
    if out.exists() and out.read_text()!=payload: raise RuntimeError('window manifest changed')
    tmp=out.with_suffix('.tmp');tmp.write_text(payload);tmp.replace(out)
    print(len(selected),pd.DataFrame(selected).groupby('route').size().to_dict(),ref)
if __name__=='__main__': main()
