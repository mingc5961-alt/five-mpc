from dataclasses import dataclass,field
import numpy as np
from .geometry import angles,angle_error

@dataclass
class Forecast:
    origin: float
    hs: np.ndarray
    centers: dict
    covariance: dict
    features: np.ndarray
    diagnostics: dict
    def at(self,name,h):
        if h<0 or h>self.hs[-1]+1e-8: raise ValueError('Forecast age outside issued horizon')
        return np.array([np.interp(h,self.hs,self.centers[name][:,j]) for j in range(3)])

class Evidence:
    """新测量到达后核对过去发布的位置，产生决策时已经兑现的历史误差特征。"""
    def __init__(self): self.archive=[]; self.realized=[]
    def observe(self,obs):
        for package in self.archive:
            lag=obs.t-package.origin
            if .5<=lag<=package.hs[-1]+1e-8:
                for name in ['P','I','CR_raw','H']:
                    if name not in package.centers: continue
                    error=float(np.linalg.norm(angle_error(angles(package.at(name,lag)),angles(obs.position))))
                    self.realized.append((obs.t,lag,name,error))
        self.archive=[p for p in self.archive if obs.t-p.origin<=p.hs[-1]]
        self.realized=[r for r in self.realized if obs.t-r[0]<=60]
    def summary(self,t):
        features=[]
        for name in ['P','I','CR_raw']:
            for low,high in [(.5,2),(2,10)]:
                data=[r for r in self.realized if r[2]==name and low<=r[1]<=high]
                vals=np.array([r[3] for r in data]); count=len(vals)
                features.extend([float(vals.mean()) if count else 0.,float(vals.std()) if count else 0.,count/100,min(t-data[-1][0],60)/60 if count else 1.])
        for low,high in [(.5,2),(2,10)]:
            vals=[r[3] for r in self.realized if r[2]=='H' and low<=r[1]<=high]
            features.extend([float(np.mean(vals)) if vals else 0.,len(vals)/100])
        return np.array(features)
    def issue(self,package): self.archive.append(package)
