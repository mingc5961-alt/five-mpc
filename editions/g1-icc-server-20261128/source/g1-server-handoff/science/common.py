"""Truth sampling used by offline scoring of completed forecasts."""
import numpy as np


def truth(world, user, times):
    q = np.asarray(times) / world.dt
    if np.any(q < 0) or np.any(q > len(world.times)-1):
        raise ValueError('query outside simulated truth horizon')
    j = np.floor(q).astype(int)
    frac = (q - j)[..., None]
    xyz = world.positions[user]
    return (1-frac)*xyz[j] + frac*xyz[np.minimum(j+1,len(xyz)-1)]
