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

