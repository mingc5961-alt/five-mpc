"""Exact F schedule replay over nine widths for each of the four feasible periods."""
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from routeclock.config import load_config
from routeclock.filters import PositionIMM
from routeclock.geometry import angles
from routeclock.radio import Radio,array_gain
from routeclock.scheduler import TokenBucket
from routeclock.world import World
from .data import BASE
from .models import HS
from .network import warm_history,interpolate

def schedule(world,start,end,period,states,origins):
    cost=.0031;rho=.05/3;bucket=TokenBucket(cost,rho)
    packages=[[(t,imm.forecast(HS)[0])] for t,imm in zip(origins,states)]
    due=np.array(origins)+period-cost
    times=[];rr=0;t=start;sensed=0.;switch=0.
    quantum=world.cfg['simulation']['tdma_quantum_s'];sw=world.cfg['simulation']['beam_switch_s']
    while t<end-1e-10:
        i=np.argmin(due);at=max(t,due[i],bucket.ready(t));can_sense=at+cost<=end
        stop=at if can_sense else end
        if stop>t:
            count=int(np.ceil((stop-t)/quantum-1e-10))
            for j in range(count):
                left=t+j*quantum;right=min(left+quantum,stop)
                data=min(right,left+sw);switch+=data-left
                if right>data:times.append((rr%world.n,data,right))
                rr+=1
        if not can_sense:break
        bucket.consume(at);t=at+cost;sensed+=cost
        states[i].observe(world.observe(i,t));packages[i].append((t,states[i].forecast(HS)[0]))
        due[i]=t+period-cost
    return packages,np.array(times),sensed,switch

def pointings(clocks,centers,t):
    ix=clocks.searchsorted(t,side='right')-1;age=t-clocks[ix]
    step=np.clip(np.floor(age/.5).astype(int),0,19)
    u=np.clip((age-HS[step])/.5,0,1)
    return centers[ix,step]*(1-u[:,None])+centers[ix,step+1]*u[:,None],age

def grid(job):
    folder=BASE/'q1_frontier';folder.mkdir(exist_ok=True)
    dest=folder/f'F_world{job:02}.csv'
    if dest.exists():return
    cfg=load_config();cfg['simulation']['sensing_budget']=.05/3
    world=World(cfg,'refine_v1/search',job);warm=180.;duration=180.;radio=Radio(cfg)
    histories=warm_history(world,warm,.0031)
    imms=[];origins=[]
    for h in histories:
        imm=PositionIMM()
        for o in h:imm.observe(o)
        imms.append(imm);origins.append(h[-1].t)
    rows=[]
    probe=warm+(np.arange(9000)+.5)*.02
    for period in [3.,5.,8.,10.]:
        packs,slots,sensed,switch=schedule(world,warm,warm+duration,period,copy.deepcopy(imms),origins)
        by_width=np.zeros((len(radio.pairs),3))
        usable=0.;unavailable=0.;no_action=0
        for uid,history in enumerate(packs):
            clocks=np.array([t for t,_ in history]);centers=np.array([p for _,p in history])
            left,right=slots[slots[:,0]==uid,1:].T
            ix=clocks.searchsorted(left,side='right')-1
            stop=np.maximum(left,np.minimum(right,clocks[ix]+period))
            usable+=sum(stop-left);unavailable+=sum(right-stop)
            ok=stop>left;dt=(stop-left)[ok];mids=(left[ok]+stop[ok])*.5
            pp,age=pointings(clocks,centers,probe)
            mask=age<period;no_action+=(~mask).sum()
            p_data,_=pointings(clocks,centers,mids)
            pred=np.vstack([pp[mask],p_data]);direction=angles(pred)
            times=np.r_[probe[mask],mids]
            xyz=interpolate(world.positions[uid],times,world.dt)
            actual=angles(xyz);split=mask.sum()
            shadow=interpolate(world.shadows[uid],mids,world.dt)
            for j,pair in enumerate(radio.pairs):
                geom=10*np.log10(np.prod(pair)/array_gain(direction,actual,pair))
                by_width[j,0]+=(geom[:split]>3).sum()+(~mask).sum()
                nom=radio.snr(p_data,direction[split:],pair)
                mcs=(radio.threshold.searchsorted(nom,side='right')-1).clip(0,len(radio.se)-1)
                snr=radio.snr(xyz[split:],direction[split:],pair,shadow)
                by_width[j,1]+=radio.goodput(snr,mcs)[0]@dt
                if pair==(32,32):by_width[j,2]+=dt[geom[split:]<=3].sum()
        assert abs(sensed+switch+usable+unavailable-duration)<1e-6
        assert sensed<=duration*.05/3+.0031+1e-8
        for pair,(risk,bits,narrow) in zip(radio.pairs,by_width,strict=True):
            rows.append(dict(world=job,method='F',width_x=pair[0],width_y=pair[1],period_s=period,
                      R=risk/(world.n*len(probe)),G_mbps=bits/duration/1e6,unavailable_probe=no_action/(world.n*len(probe)),
                      sensing_s=sensed,switch_s=switch,usable_s=usable,unavailable_s=unavailable,narrow_good_data_s=narrow,
                      narrow_share=float(pair==(32,32))))
    tmp=dest.with_suffix('.part');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(dest)
    print('completed',job,flush=True)
if __name__=='__main__':
    with ProcessPoolExecutor(3,mp_context=mp.get_context('fork')) as ex:list(ex.map(grid,range(10)))
    files=sorted((BASE/'q1_frontier').glob('F_world*.csv'))
    frame=pd.concat([pd.read_csv(p) for p in files]);assert len(frame)==360
    frame.to_csv(BASE/'q1_frontier/F_scores.csv',index=False)
