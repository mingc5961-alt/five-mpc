import numpy as np
from .data import BASE
from routeclock.world import Observation

def records(path):
    with np.load(path) as data:
        history = [Observation(float(t),p,c,float(v),float(s),None)
                   for t,p,c,v,s in zip(data['times'],data['positions'],data['cov'],data['radial'],data['radial_std'],strict=True)]
        origins,truths,times = data['origins'],data['truth'],data['times']
    for origin, truth in zip(origins,truths,strict=True):
        idx = times.searchsorted(origin,side='right')
        assert times[idx-1] == origin
        yield history[:idx],truth

def paths(split):
    return sorted((BASE/'observations'/split).glob('*.npz'))
