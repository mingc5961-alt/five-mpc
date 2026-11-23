"""Retrospective scoring of issued beams; never consulted by the scheduler."""
import numpy as np
from .geometry import angles
from .radio import array_gain

def score_holds(world,forecasts,frames,start,end):
    archive={(p['uav'],p['origin']):p for p in forecasts}
    if not frames or abs(frames[0]['t']-start)>1e-7:
        raise ValueError('Missing initial service choices')
    rows=[]
    for uid in range(world.n):
        spans=[]
        for frame in frames:
            choice=frame['choices'][uid]
            key=(choice['origin'],tuple(choice['pair'] or ()))
            if not spans or spans[-1][1]!=key:spans.append((frame['t'],key))
        for j,(begin,(origin,pair)) in enumerate(spans):
            finish=spans[j+1][0] if j+1<len(spans) else end
            if not pair or finish<=begin:continue
            p=archive[(uid,origin)];ts=np.r_[np.arange(begin,finish,world.dt),finish]
            ix=np.minimum((ts/world.dt).astype(int),len(world.times)-2);u=(ts-world.times[ix])/world.dt
            truth=world.positions[uid,ix]*(1-u[:,None])+world.positions[uid,ix+1]*u[:,None]
            pred=np.stack([np.interp(ts-origin,p['h'],x) for x in np.array(p['center']).T],axis=1)
            loss=-10*np.log10(array_gain(angles(pred),angles(truth),pair)/np.prod(pair))
            rows.append((finish-begin,pair==(32,32),bool(np.any(loss>3)),bool(np.any(ts-origin>p['h'][-1]+1e-8))))
    narrow=[r for r in rows if r[1]]
    return dict(hold_segments=len(rows),hold_failures=sum(r[2] for r in rows),expired_segments=sum(r[3] for r in rows),
                narrow_segments=len(narrow),narrow_failures=sum(r[2] for r in narrow),
                narrow_hold_s=float(np.mean([r[0] for r in narrow])) if narrow else None,
                narrow_share=sum(r[0] for r in narrow)/((end-start)*world.n))



def score_selected_coverage(world,forecasts,frames):
    """Score each distinct selected (user, issue, deadline) once, on the truth grid."""
    from .geometry import angle_error
    requests = {}
    for frame in frames:
        for uid,choice in enumerate(frame['choices']):
            if choice['pair']:
                requests.setdefault((uid,choice['origin']),set()).add(choice['interval'])
    counts = {}
    for p in forecasts:
        uid,origin=p['uav'],p['origin']
        deadlines=requests.get((uid,origin))
        if not deadlines or p['envelope_rad'] is None:continue
        hs=np.asarray(p['h'])
        ages=np.unique(np.r_[np.arange(0,max(deadlines),world.dt),list(deadlines)])
        ts=origin+ages
        ix=np.minimum((ts/world.dt).astype(int),len(world.times)-2)
        u=(ts-world.times[ix])/world.dt
        truth=world.positions[uid,ix]*(1-u[:,None])+world.positions[uid,ix+1]*u[:,None]
        pred=np.column_stack([np.interp(ages,hs,x) for x in np.asarray(p['center']).T])
        err=np.abs(angle_error(angles(pred),angles(truth)))
        env=np.asarray(p['envelope_rad'])[np.searchsorted(hs,ages)]
        for delta in deadlines:
            ratio=(p.get('deadline_ratios') or {}).get(delta,1.)
            hit=bool((err[ages<=delta]<=env[ages<=delta]*ratio).all())
            row=counts.setdefault(str(delta),dict(selected=0,covered=0))
            row['selected']+=1
            row['covered']+=int(hit)
    return counts
