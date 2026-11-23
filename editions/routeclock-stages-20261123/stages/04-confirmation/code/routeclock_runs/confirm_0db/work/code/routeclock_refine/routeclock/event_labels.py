import numpy as np
from .geometry import unit
from .world import GROUPS

def event_at(world,uid,t):
    """Evaluation-only realized geometry and generator event time, never a model input."""
    curve=world.references[uid]
    def residual(at):
        pos=world.truth(uid,at)[0]
        segments=np.diff(curve,axis=0)
        ratio=np.clip(np.sum((pos-curve[:-1])*segments,axis=1)/np.maximum(1e-12,np.sum(segments**2,axis=1)),0,1)
        return float(np.min(np.linalg.norm(pos-curve[:-1]-ratio[:,None]*segments,axis=1)))
    error=residual(t);group=world.groups[uid]
    recent=[(s,e) for s,e in world.detour_windows[uid] if 0<=t-s and t<=e+10]
    if recent and error>3:label='detour_off_path'
    elif recent and t>=recent[-1][1] and error<=3:label='post_detour_on_path'
    else:
        v=world.truth(uid,t)[1];future=world.truth(uid,t+2)[1]
        turn=np.degrees(np.arccos(np.clip(unit(v)@unit(future),-1,1)))
        label='on_path_upcoming_turn' if group!=GROUPS[-1] and error<=3 and turn>=15 and min(np.linalg.norm(v),np.linalg.norm(future))>.5 else 'other'
    return label,error

