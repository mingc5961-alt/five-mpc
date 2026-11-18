from dataclasses import dataclass
import numpy as np
from .geometry import angles

@dataclass
class Action:
    pair: tuple
    interval: float
    rate: float
    loss_db: float

def frontier(package,center,envelope,radio,cfg,action_head=None,features=None):
    actions=[]; intervals=cfg['simulation']['intervals_s']
    positions=package.centers[center]; pointing=angles(positions)
    for pair in radio.pairs:
        loss=np.array([radio.worst_loss(a,e,pair) for a,e in zip(pointing,envelope)])
        valid=loss<=cfg['radio']['geometric_loss_threshold_db']
        if action_head is not None:
            if action_head.target=='duration':
                lower=float(action_head.lower(action_head.rows(features,package.hs,pair,[0]))[0]); valid=package.hs<=max(0,lower)
            else:
                lower=action_head.lower(action_head.rows(features,package.hs,pair,package.hs)); valid=lower>=radio.threshold[0]
        first=np.flatnonzero(~valid); last_h=package.hs[first[0]-1] if len(first) and first[0]>0 else (0 if len(first) else package.hs[-1])
        for delta in intervals:
            if delta>last_h+1e-8:continue
            sel=package.hs<=delta
            predicted=radio.snr(positions[sel],pointing[sel],pair)-np.minimum(loss[sel],40)
            rates=[radio.goodput(s,radio.mcs(s))[0] for s in predicted]
            actions.append(Action(pair,float(delta),float(np.mean(rates)),float(loss[sel].max())))
    # Keep all candidates here: restrictions in an ablation must precede pruning.
    return actions

def choose_actions(frontiers,cost,budget,mode='value',width_only=False,interval_only=False,fixed_interval=1.,fixed_pair=(16,16),trace=None):
    if mode=='uniform':
        selection=[None]*len(frontiers)
        for axis in [32,16,8]:
            selection=[max([a for a in fs if a.pair==(axis,axis)],key=lambda a:a.interval,default=None) for fs in frontiers]
            load=sum(cost/a.interval if a else 1. for a in selection)
            if load<=budget:break
        return selection
    candidates=[]
    for fs in frontiers:
        fs=[a for a in fs if (not width_only or a.interval==fixed_interval) and (not interval_only or a.pair==fixed_pair)]
        fs=[a for a in fs if not any(b.rate>=a.rate and b.interval>=a.interval and (b.rate>a.rate or b.interval>a.interval) for b in fs)]
        candidates.append(fs)
    chosen=[max(fs,key=lambda a:a.rate) if fs else None for fs in candidates]
    if trace is not None:
        trace.update(raw=[len(fs) for fs in frontiers],pareto=[len(fs) for fs in candidates],
                     available_widths=[len({a.pair for a in fs}) for fs in candidates],
                     budget_moves=[],tested_width_changes=0,accepted_width_changes=0)
    def load(a):return cost/a.interval if a is not None and a.interval>0 else 1.
    if mode=='edf':return chosen
    while sum(load(a) for a in chosen)>budget+1e-12:
        options=[]
        for i,(current,fs) in enumerate(zip(chosen,candidates)):
            if current is None:continue
            for a in fs:
                saved=load(current)-load(a)
                if saved>1e-12:
                    score=(current.rate-a.rate)/saved if mode=='value' else -saved
                    options.append((score,i,a))
        if not options:break
        _,i,a=min(options,key=lambda x:x[0])
        if trace is not None:
            trace['budget_moves'].append((i,chosen[i].pair,a.pair))
        chosen[i]=a
    if mode=='value':
        # Coordinate ascent on the shared-airtime objective, including sensing
        # overhead even when the hard budget is not saturated.
        def value(selection):
            total_load=sum(load(a) for a in selection)
            return max(0,1-total_load)*sum(a.rate if a else 0 for a in selection)/max(1,len(selection))
        for _ in range(100):
            base=value(chosen);best=None
            for i,fs in enumerate(candidates):
                for a in fs:
                    trial=chosen.copy();trial[i]=a
                    if sum(load(b) for b in trial)>budget+1e-12:continue
                    if trace is not None and chosen[i] is not None and a.pair!=chosen[i].pair:
                        trace['tested_width_changes']+=1
                    improvement=value(trial)-base
                    if improvement>1e-6 and (best is None or improvement>best[0]):best=(improvement,i,a)
            if best is None:break
            _,i,a=best
            if trace is not None and chosen[i] is not None and a.pair!=chosen[i].pair:
                trace['accepted_width_changes']+=1
            chosen[i]=a
    return chosen

class TokenBucket:
    def __init__(self,cost,budget):self.cost=cost;self.budget=budget;self.tokens=cost;self.t=0.
    def ready(self,t):
        self.tokens=min(self.cost,self.tokens+max(0,t-self.t)*self.budget);self.t=t
        return t+max(0,self.cost-self.tokens)/self.budget
    def consume(self,t):
        self.ready(t)
        if self.tokens+1e-10<self.cost:raise RuntimeError('Sensing budget violated')
        self.tokens-=self.cost
