"""Interacting multiple-model Bayesian filters, with irregular-time mixing."""
import copy
import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp
from functools import lru_cache
from .geometry import unit

def moment(weights,means,covs):
    mean=np.einsum('k,ki->i',weights,means); d=means-mean
    cov=np.einsum('k,kij->ij',weights,covs+np.einsum('ki,kj->kij',d,d))
    return mean,(cov+cov.T)/2

def mix(states,covs,weights,trans):
    joint=weights[:,None]*trans; prior=joint.sum(axis=0); prior=np.maximum(prior,1e-14); prior/=prior.sum()
    pairs=[moment(joint[:,j]/max(joint[:,j].sum(),1e-14),states,covs) for j in range(len(prior))]
    return np.array([x[0] for x in pairs]),np.array([x[1] for x in pairs]),prior

def update_linear(x,P,z,H,R):
    residual=z-H@x; S=H@P@H.T+R; inv=np.linalg.inv(S); K=P@H.T@inv
    xx=x+K@residual; J=np.eye(len(x))-K@H
    PP=J@P@J.T+K@R@K.T
    loglik=-.5*(len(z)*np.log(2*np.pi)+np.linalg.slogdet(S)[1]+residual@inv@residual)
    return xx,PP,loglik,float(residual@inv@residual)

@lru_cache(maxsize=1024)
def position_transitions(dt):
    matrices=[]
    for w in (0.,.18,-.18):
        A=np.zeros((6,6)); A[:3,3:]=np.eye(3); A[3,4]=-w; A[4,3]=w
        matrices.append(expm(A*dt))
    generator=np.full((3,3),.04); np.fill_diagonal(generator,-.08)
    return expm(generator*dt), np.array(matrices)

class PositionIMM:
    """保留多个运动假设，预测后用观测更新其权重，再按矩匹配合并位置及协方差。"""
    def __init__(self):
        self.t=None; self.weights=np.array([.8,.1,.1]); self.Q=np.full((3,3),.04); np.fill_diagonal(self.Q,-.08)
        self.noise=1.0; self.nis=0.0
    def _advance(self,dt):
        transition,matrices=position_transitions(dt)
        self.x,self.P,self.weights=mix(self.x,self.P,self.weights,transition)
        for j,w in enumerate([0,.18,-.18]):
            F=matrices[j]; G=np.vstack([.5*dt*dt*np.eye(3),dt*np.eye(3)])
            self.x[j]=F@self.x[j]; self.P[j]=F@self.P[j]@F.T+self.noise*(G@G.T)+np.eye(6)*1e-8
    def observe(self,obs):
        if self.t is None:
            self.x=np.tile(np.r_[obs.position,unit(obs.position)*obs.radial_velocity],(3,1))
            self.P=np.tile(np.eye(6),(3,1,1)); self.P[:,:3,:3]=obs.covariance; self.P[:,3:,3:]*=25; self.t=obs.t; return
        self._advance(obs.t-self.t)
        H=np.zeros((4,6)); H[:3,:3]=np.eye(3); H[3,3:]=unit(obs.position)
        R=np.zeros((4,4)); R[:3,:3]=obs.covariance; R[3,3]=obs.radial_std**2
        updated=[update_linear(x,P,np.r_[obs.position,obs.radial_velocity],H,R) for x,P in zip(self.x,self.P)]
        self.x=np.array([u[0] for u in updated]); self.P=np.array([u[1] for u in updated]); self.nis=float(np.dot(self.weights,[u[3] for u in updated]))
        logs=np.log(np.maximum(self.weights,1e-14))+np.array([u[2] for u in updated]); self.weights=np.exp(logs-logsumexp(logs))
        self.noise=float(np.clip(.9*self.noise+.1*self.nis/4,.05,16)); self.t=obs.t
    def state(self): return moment(self.weights,self.x,self.P)
    def forecast(self,hs):
        f=copy.deepcopy(self); out=[]; cov=[]; prev=0
        for h in hs:
            if h>prev: f._advance(float(h-prev))
            m,P=f.state(); out.append(m[:3]); cov.append(P[:3,:3]); prev=h
        return np.array(out),np.array(cov)

@lru_cache(maxsize=4096)
def ou_matrices(dt,tau,diffusion):
    A=np.array([[0.,1.],[0.,-1/tau]]); D=np.diag([0.,diffusion**2]); block=np.block([[A,D],[np.zeros((2,2)),-A.T]])
    e=expm(block*dt); F=e[:2,:2]; Q=e[:2,2:]@F.T
    return F,(Q+Q.T)/2

class ProgressIMM:
    """在沿路线的弧长坐标上估计进度与速度，让同一路线可以按不同节奏执行。"""
    def __init__(self,cfg,s=0,speed=0,t=0):
        self.cfg=cfg; self.t=t; self.cruise=max(speed,.5); self.x=np.tile([s,speed],(3,1)).astype(float)
        self.P=np.tile(np.diag([1.,2.]),(3,1,1)); self.weights=np.array([.8,.15,.05]); self.speed_history=[]
        self.generator=np.array(cfg['mode_transition_generator_per_s'])
    def advance(self,dt):
        self.x,self.P,self.weights=mix(self.x,self.P,self.weights,expm(self.generator*dt))
        for j,(tau,diff,mult) in enumerate(zip(self.cfg['progress_ou_tau_s'],self.cfg['progress_ou_diffusion_m_s_per_sqrt_s'],self.cfg['progress_mean_speed_factors'])):
            F,Q=ou_matrices(round(dt,8),tau,diff); target=self.cruise*mult
            b=np.array([target*(dt-tau*(1-np.exp(-dt/tau))),target*(1-np.exp(-dt/tau))])
            self.x[j]=F@self.x[j]+b; self.P[j]=F@self.P[j]@F.T+Q
    def observe(self,s,speed,var,t):
        self.advance(t-self.t)
        self.speed_history=[(tt,v) for tt,v in self.speed_history if t-tt<=20]
        if speed>.5: self.speed_history.append((t,speed))
        if self.speed_history: self.cruise=float(np.median([v for _,v in self.speed_history]))
        updates=[update_linear(x,P,np.array([s,speed]),np.eye(2),np.diag([max(var,.1),1.])) for x,P in zip(self.x,self.P)]
        self.x=np.array([u[0] for u in updates]); self.P=np.array([u[1] for u in updates])
        logs=np.log(np.maximum(self.weights,1e-14))+np.array([u[2] for u in updates]); self.weights=np.exp(logs-logsumexp(logs)); self.t=t
    def forecast(self,hs):
        f=copy.deepcopy(self); result=[]; prev=0
        for h in hs:
            if h>prev: f.advance(float(h-prev))
            result.append((f.weights.copy(),f.x.copy(),f.P.copy())); prev=h
        return result
