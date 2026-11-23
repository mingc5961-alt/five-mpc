"""Evaluate independent radio array candidates together using the frozen radio model."""
import numpy as np
from refine.network import HS, QUERY_H
from routeclock.geometry import angles
from routeclock.scheduler import Action

HORIZONS=QUERY_H[1:]
MASK=HS[None,:]<=HORIZONS[:,None]
COUNTS=MASK.sum(axis=1)

def frontier(radio,center,scale,factors):
    direction=angles(center)
    env=np.maximum(scale,np.vstack([scale[:1],scale[:-1]]))
    bounds=(env[None,:,:]*factors[:,None,None]).reshape(-1,2)
    directions=np.tile(direction,(len(HORIZONS),1))
    pairs=np.asarray(radio.pairs).T[:,:,None]
    nominal=radio.snr(center,direction,pairs)
    mcs=np.searchsorted(radio.threshold,nominal,side='right').clip(1,len(radio.se))-1
    losses=radio.worst_losses(directions,bounds,pairs).reshape(len(radio.pairs),len(HORIZONS),len(HS))
    good,_=radio.goodput(nominal[:,None,:]-np.minimum(losses,40),mcs[:,None,:])
    worst=np.where(MASK,losses,0.).max(axis=2)
    feasible=np.logical_and.accumulate(worst<=3.,axis=1)
    rates=np.sum(good*MASK,axis=2)/COUNTS
    return [Action(radio.pairs[i],float(HORIZONS[j]),float(rates[i,j]),float(worst[i,j])) for i,j in zip(*np.nonzero(feasible))]
