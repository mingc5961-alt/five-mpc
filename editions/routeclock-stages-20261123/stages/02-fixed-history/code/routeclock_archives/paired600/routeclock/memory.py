from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from .geometry import unit
from .filters import ProgressIMM,moment

@dataclass
class Fragment:
    points: np.ndarray
    closed: bool=False
    visits: int=1
    def arc(self): return np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
    def at(self,s):
        arc=self.arc(); length=arc[-1]
        if self.closed and length>0: s=s%length
        if s>length:
            tangent=unit(self.points[-1]-self.points[-2]); return self.points[-1]+(s-length)*tangent,0.,tangent
        if s<0: return self.points[0]+s*unit(self.points[1]-self.points[0]),0.,unit(self.points[1]-self.points[0])
        j=min(max(np.searchsorted(arc,s,side='right')-1,0),len(arc)-2)
        tangent=unit(self.points[j+1]-self.points[j]); return self.points[j]+(s-arc[j])*tangent,1.,tangent
    def resample(self,spacing):
        if len(self.points)<3:return
        arc=self.arc(); grid=np.r_[np.arange(0,arc[-1],spacing),arc[-1]]
        # Local smoothing with fixed endpoints; only real accepted observations enter.
        p=self.points.copy(); p[1:-1]=.25*p[:-2]+.5*p[1:-1]+.25*p[2:]
        self.points=np.c_[*[np.interp(grid,arc,p[:,j]) for j in range(3)]]

class RouteMemory:
    def __init__(self,cfg):
        self.cfg=cfg; self.fragments=[]; self.last=None; self.matches=[]; self.active=None; self.s=0.; self.progress=None
        self.innovation=0.; self.heading_error=0.; self.entropy=0.; self.new_points=0; self.revisits=0; self.diagnostic={}
        self.connections={}
    def update(self,obs,velocity):
        p=obs.position; speed=np.linalg.norm(velocity); variance=max(np.trace(obs.covariance)/3,.05)
        if self.last is None:
            self.last=obs; self.diagnostic={'innovation':0.,'support':0.}; return
        dt=obs.t-self.last.t
        if not self.fragments:
            if np.linalg.norm(p-self.last.position)<.5: self.last=obs; return
            self.fragments=[Fragment(np.vstack([self.last.position,p]))]; self.active=0; self.s=self.fragments[0].arc()[-1]
            self.progress=ProgressIMM(self.cfg,self.s,speed,obs.t); self.last=obs; self.matches=[(0,self.s,1.)]; return
        candidates=[]
        for fi,frag in enumerate(self.fragments):
            a=frag.points[:-1]; d=np.diff(frag.points,axis=0); lengths=np.linalg.norm(d,axis=1); valid=lengths>1e-6
            u=np.clip(np.einsum('ij,ij->i',p-a,d)/np.maximum(lengths**2,1e-12),0,1)
            projected=a+u[:,None]*d; dist=np.linalg.norm(projected-p,axis=1); arc=frag.arc()
            for j in np.argsort(dist)[:5]:
                if not valid[j]:continue
                tang=d[j]/lengths[j]; heading=1-np.dot(unit(velocity),tang) if speed>.5 else 0
                s=arc[j]+u[j]*lengths[j]; prior=0.
                if fi==self.active:
                    ds=s-self.s
                    if frag.closed: ds=(ds+arc[-1]/2)%arc[-1]-arc[-1]/2
                    prior=((ds-speed*dt)/max(3,2*speed*dt))**2
                score=dist[j]**2/(variance+1)+3*heading+prior
                candidates.append((score,fi,s,float(dist[j]),float(heading),projected[j]))
        candidates.sort(key=lambda x:x[0]); best=candidates[0]
        # This residual is measured before any map change.
        self.innovation=best[3]/np.sqrt(variance+1); self.heading_error=best[4]
        accepted=best[3]<=self.cfg['matching_position_gate_mahalanobis']*np.sqrt(variance+1) and (speed<.5 or best[4]<1)
        prev_active=self.active; prev_s=self.s
        frag=self.fragments[prev_active]
        displacement=np.linalg.norm(p-self.last.position)
        if accepted:
            selected=[]
            for c in candidates:
                if c[0]>best[0]+9:break
                if all(c[1]!=q[1] or abs(c[2]-q[2])>3 for q in selected): selected.append(c)
                if len(selected)>=self.cfg['matching_max_hypotheses']:break
            weights=softmax(-.5*np.array([c[0] for c in selected])); self.matches=[(c[1],c[2],float(w)) for c,w in zip(selected,weights)]
            self.entropy=float(-np.sum(weights*np.log(np.maximum(weights,1e-12))))
            self.active=best[1]; self.s=best[2]; chosen=self.fragments[self.active]
            if self.active==prev_active and self.s<prev_s-10: self.revisits+=1; chosen.visits+=1
            # Extend a continuous terminal even if its endpoint is still within the gate.
            end=chosen.arc()[-1]; tangent=unit(chosen.points[-1]-chosen.points[-2])
            if not chosen.closed and self.s>end-1 and np.dot(p-chosen.points[-1],tangent)>.7:
                chosen.points=np.vstack([chosen.points,p]); self.s=chosen.arc()[-1]; self.new_points+=1
            else:
                idx=int(np.argmin(np.linalg.norm(chosen.points-p,axis=1)))
                chosen.points[idx]=.95*chosen.points[idx]+.05*p
        elif displacement>.5:
            # Only append at a continuous terminal. Mid-route departure starts a new fragment.
            end=frag.arc()[-1]
            if not frag.closed and end-prev_s<max(3,displacement*1.5):
                frag.points=np.vstack([frag.points,p]); self.s=frag.arc()[-1]; self.new_points+=1
            else:
                self.fragments.append(Fragment(np.vstack([self.last.position,p]))); self.active=len(self.fragments)-1; self.s=displacement
            self.matches=[(self.active,self.s,1.)]; self.entropy=0.
        chosen=self.fragments[self.active]
        # A mid-route departure is a separate fragment, not a continuation of
        # the old route's terminal. Only observed terminal transitions connect.
        old_terminal=self.fragments[prev_active].arc()[-1]
        if self.active!=prev_active and not self.fragments[prev_active].closed and old_terminal-prev_s<=max(1.,speed*dt*1.5):
            key=(prev_active,self.active)
            count,entry=self.connections.get(key,(0,self.s))
            self.connections[key]=(count+1,(entry*count+self.s)/(count+1))
        # Closure needs the observed suffix to approach the start in its forward direction.
        if not chosen.closed and len(chosen.points)>=12 and chosen.arc()[-1]>25:
            startdir=unit(chosen.points[2]-chosen.points[0]); enddir=unit(chosen.points[-1]-chosen.points[-3])
            near=np.linalg.norm(p-chosen.points[0])<max(2,2*np.sqrt(variance))
            preceding=np.linalg.norm(chosen.points[-3]-chosen.points[0])>np.linalg.norm(p-chosen.points[0])
            if near and np.dot(startdir,enddir)>.75 and preceding and prev_s>chosen.arc()[-1]*.7:
                chosen.points=np.vstack([chosen.points,chosen.points[0]]); chosen.closed=True; self.revisits+=1
        if self.new_points>=self.cfg['rebuild_after_new_accepted_points']:
            before=chosen.arc()[-1]; chosen.resample(self.cfg['resample_spacing_m']); self.s*=chosen.arc()[-1]/max(before,1e-9); self.new_points=0
        if self.matches:
            shift=self.s-self.matches[0][1]
            self.matches=[(fi,s+shift if fi==self.active else s,w) for fi,s,w in self.matches]
        if self.progress is None or self.active!=prev_active:
            self.progress=ProgressIMM(self.cfg,self.s,speed,obs.t)
        else:
            measured_s=self.s
            if chosen.closed:
                L=chosen.arc()[-1]; expected=float(self.progress.weights@self.progress.x[:,0])+speed*dt
                measured_s+=round((expected-measured_s)/L)*L
            tangent=chosen.at(self.s)[2]
            self.progress.observe(measured_s,max(0,float(velocity@tangent)),variance,obs.t)
        self.last=obs
        self.diagnostic={'innovation':self.innovation,'heading':self.heading_error,'match_entropy':self.entropy,'fragments':len(self.fragments),'revisits':self.revisits,'active':self.active,'s':self.s,'closed':chosen.closed}
    def map_distribution(self,fi,s,depth=0):
        frag=self.fragments[fi];length=frag.arc()[-1]
        outgoing=[(target,count,entry) for (source,target),(count,entry) in self.connections.items() if source==fi]
        if s>length and not frag.closed and outgoing and depth<8:
            total=sum(x[1] for x in outgoing);result=[]
            for target,count,entry in outgoing:
                destination=self.fragments[target].at(entry)[0];terminal=frag.points[-1]
                bridge=np.linalg.norm(destination-terminal);remaining=s-length
                if bridge>1e-9 and remaining<bridge:
                    result.append((terminal+(destination-terminal)*remaining/bridge,1.,count/total))
                else:
                    for p,support,w in self.map_distribution(target,entry+remaining-bridge,depth+1):result.append((p,support,w*count/total))
            return result
        p,support,_=frag.at(s);return [(p,support,1.)]
    def forecast(self,hs,point=False):
        if self.progress is None:
            p=self.last.position; return np.tile(p,(len(hs),1)),np.tile(self.last.covariance,(len(hs),1,1)),np.zeros(len(hs)),np.zeros(len(hs)),np.tile([1.,0.,0.],(len(hs),1))
        states=self.progress.forecast(hs); means=[]; covs=[]; supports=[]; stds=[]; modes=[]
        for w,x,P in states:
            m,C=moment(w,x,P); stds.append(np.sqrt(max(C[0,0],0))); modes.append(w)
            if point:
                # The ablation collapses within-mode AND between-mode progress
                # uncertainty, while retaining separate geometric match branches.
                x=np.tile(m,(3,1));P=np.zeros_like(P);stds[-1]=0.
            cloud=[]; weights=[]; supp=[]
            matches=self.matches or [(self.active,self.s,1.)]
            for fi,s,wm in matches:
                for mode in range(3):
                    offsets=[(0,1.)] if point else [(0,2/3),(-np.sqrt(3*max(P[mode,0,0],0)),1/6),(np.sqrt(3*max(P[mode,0,0],0)),1/6)]
                    for ds,wp in offsets:
                        current_mean=float(self.progress.weights@self.progress.x[:,0])
                        # Each matching hypothesis, including same-fragment crossings,
                        # owns its projected coordinate. Lift it near the active lap.
                        base=current_mean
                        if fi==self.active and self.fragments[fi].closed:
                            L=self.fragments[fi].arc()[-1];s_lift=s+round((base-s)/L)*L
                        else:s_lift=s
                        delta=s_lift-base
                        for pp,supported,branch_weight in self.map_distribution(fi,x[mode,0]+ds+delta):
                            cloud.append(pp); weights.append(wm*w[mode]*wp*branch_weight); supp.append(supported)
            weights=np.array(weights); weights/=weights.sum(); cloud=np.array(cloud); mm=weights@cloud; dd=cloud-mm
            means.append(mm); covs.append(np.einsum('k,ki,kj->ij',weights,dd,dd)+self.last.covariance)
            supports.append(float(weights@np.array(supp)))
        return np.array(means),np.array(covs),np.array(supports),np.array(stds),np.array(modes)
