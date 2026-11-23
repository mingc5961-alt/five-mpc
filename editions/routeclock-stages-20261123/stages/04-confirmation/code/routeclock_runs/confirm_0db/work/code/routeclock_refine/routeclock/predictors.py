import numpy as np
from scipy.special import softmax,expit
from scipy.optimize import minimize
from .filters import PositionIMM
from .memory import RouteMemory
from .evidence import Evidence,Forecast
from .geometry import angles,angle_error,angular_jacobian

class HarmonicMemory:
    """Period/drift-grid Bayesian linear state-space experts, updated causally."""
    def __init__(self,cfg):
        grid=cfg['period_grid_s'];periods=np.arange(grid['start'],grid['stop']+.01,grid['step'])
        self.periods=np.repeat(periods,len(cfg['drift_std_m_per_sqrt_s']));self.drift=np.tile(cfg['drift_std_m_per_sqrt_s'],len(periods))
        self.beta=None;self.logw=np.zeros(len(self.periods));self.t=None;self.origin=None;self.residual=0.
    def design(self,t):
        return np.stack([np.ones(len(self.periods)),np.full(len(self.periods),t),*[f(2*np.pi*k*t/self.periods) for k in range(1,4) for f in [np.sin,np.cos]]],axis=1)
    def observe(self,obs):
        if self.beta is None:
            self.origin=obs.t;self.t=obs.t;self.beta=np.zeros((len(self.periods),8,3));self.beta[:,0]=obs.position
            self.P=np.tile(np.diag([1.,1.]+[25.]*6),(len(self.periods),1,1));return
        dt=obs.t-self.t;X=self.design(obs.t-self.origin)
        self.P+=dt*self.drift[:,None,None]**2*np.diag([1.,.0001]+[1.]*6)[None]
        pred=np.einsum('ki,kij->kj',X,self.beta);res=obs.position-pred
        px=np.einsum('kij,kj->ki',self.P,X);S=np.einsum('ki,ki->k',X,px)+np.trace(obs.covariance)/3
        K=px/S[:,None];self.beta+=K[:,:,None]*res[:,None,:];self.P-=np.einsum('ki,kj->kij',K,px)
        self.logw+=-.5*(3*np.log(S)+np.sum(res*res,axis=1)/S);self.logw-=self.logw.max()
        w=softmax(self.logw);self.residual=float(w@np.mean(res*res,axis=1));self.t=obs.t
    def forecast(self,hs):
        w=softmax(self.logw);out=[]
        for h in hs:out.append(np.einsum('k,ki,kij->j',w,self.design(self.t-self.origin+h),self.beta))
        return np.array(out),np.array([self.residual,w@self.periods,np.sqrt(w@np.square(self.periods-w@self.periods)),-np.sum(w*np.log(w+1e-15))])

def harmonic(history,hs,step=1):
    times=np.array([o.t for o in history]); y=np.array([o.position for o in history]); t=times-times[-1]
    periods=np.arange(18,60.01,step); predictions=[]; errors=[]
    def design(x,T):return np.c_[np.ones(len(x)),x,*[f(2*np.pi*k*x/T) for k in range(1,4) for f in [np.sin,np.cos]]]
    for T in periods:
        X=design(t,T); coef=np.linalg.solve(X.T@X+np.diag([1e-6,.1]+[.1]*6),X.T@y)
        errors.append(np.mean(np.square(X@coef-y))); predictions.append(design(hs,T)@coef)
    errors=np.array(errors); w=softmax(-(errors-errors.min())/max(np.median(errors),.1))
    prediction=np.einsum('k,kij->ij',w,np.array(predictions))
    return prediction,np.array([errors.min(),w@periods,np.sqrt(w@np.square(periods-w@periods)),-np.sum(w*np.log(w+1e-15))])

class Gate:
    """16-feature logistic gate, optimized against final angular position error."""
    def __init__(self,seed=7,mask=None):self.seed=seed; self.mask=mask
    def fit(self,X,p,i,y,l2=.001):
        self.mean=X.mean(axis=0); self.std=np.maximum(X.std(axis=0),1e-5)
        Z=(X-self.mean)/self.std
        if self.mask is not None:Z=Z*self.mask
        Z=np.c_[np.ones(len(Z)),Z]
        target=angles(y)
        def objective(beta):
            w=expit(Z@beta); pred=w[:,None]*p+(1-w[:,None])*i
            err=angle_error(angles(pred),target)
            loss=np.mean(np.sum(err**2,axis=1))+l2*np.sum(beta[1:]**2)
            jac=np.array([angular_jacobian(pp) for pp in pred]); direction=np.einsum('nij,nj->ni',jac,p-i)
            dw=2*np.sum(err*direction,axis=1)*w*(1-w)/len(w)
            grad=Z.T@dw; grad[1:]+=2*l2*beta[1:]
            return loss,grad
        x0=np.random.default_rng(self.seed).normal(0,.01,Z.shape[1])
        result=minimize(objective,x0,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
        self.beta=result.x; self.fit_status={'success':bool(result.success),'iterations':result.nit,'loss':float(result.fun)}
        return self
    def weights(self,X):
        Z=(X-self.mean)/self.std
        if self.mask is not None:Z=Z*self.mask
        return expit(np.c_[np.ones(len(Z)),Z]@self.beta)
    def predict(self,X,p,i):
        g=self.weights(X); return g[:,None]*p+(1-g[:,None])*i

class Tracker:
    def __init__(self,cfg,chronos=None,point_progress=False,no_archive=False):
        self.cfg=cfg; self.imm=PositionIMM(); self.memory=RouteMemory(cfg['models']['route']); self.evidence=Evidence()
        self.harmonic=HarmonicMemory(cfg['models']['harmonic'])
        self.history=[]; self.snapshots=[]; self.chronos=chronos; self.point_progress=point_progress; self.no_archive=no_archive
    def observe(self,obs):
        self.evidence.observe(obs) # before state/map update
        self.imm.observe(obs); state,cov=self.imm.state()
        self.memory.update(obs,state[3:]); self.history.append(obs)
        self.harmonic.observe(obs)
    def issue(self,include_chronos=False):
        hs=np.arange(0,10.001,.5); obs=self.history[-1]
        I,Icov=self.imm.forecast(hs)
        P,Pcov,support,pstd,modes=self.memory.forecast(hs,self.point_progress)
        H,hdiag=self.harmonic.forecast(hs)
        archive=np.zeros(28) if self.no_archive else self.evidence.summary(obs.t)
        centers={'I':I,'P':P,'H':H}; covs={'I':Icov,'P':Pcov}
        diagnostics={'support':support,'progress_std':pstd,'modes':modes,'memory':self.memory.diagnostic,'harmonic':hdiag,'archive':archive}
        recent_times=[o.t for o in self.history[-9:]]
        diagnostics["recent_spacing_s"]=(recent_times[-1]-recent_times[0])/max(1,len(recent_times)-1)
        # Snapshots are fixed at issue-time, never retrospectively refitted.
        package=Forecast(obs.t,hs,centers,covs,np.empty((len(hs),0)),diagnostics)
        self.snapshots.append(package)
        if include_chronos:
            if self.chronos is None:raise RuntimeError('Chronos adapter is required for C/CR; no substitution allowed')
            for name,route in [('C',False),('CR_raw',True)]:
                if include_chronos is True or include_chronos==name:
                    centers[name],covs[name]=self.chronos.predict(self.history,self.snapshots,P,Pcov,support,route)
        package.features=self.features(package,'P')
        self.evidence.issue(package)
        return package
    def features(self,package,expert='P'):
        hs=package.hs; d=package.diagnostics; archive=d['archive'].copy(); obs=self.history[-1]; state,cov=self.imm.state()
        if expert!='CR_raw':archive[16:24]=0 # deployment I/RC never receives uncomputed Chronos errors
        delta=angle_error(angles(package.centers[expert]),angles(package.centers['I']))
        rows=[]
        erridx=0 if expert=='P' else 16
        for j,h in enumerate(hs):
            sensitivity=np.linalg.norm(angular_jacobian(package.centers['P'][j]))
            gate=[h/10,self.memory.innovation,self.memory.heading_error,self.memory.entropy,d['progress_std'][j]/10,d['support'][j],archive[erridx],archive[erridx+4],archive[8],archive[12],archive[erridx+2],archive[erridx+6],np.linalg.norm(delta[j]),d['modes'][j,2],d['modes'][j,1],sensitivity]
            local=[np.linalg.norm(state[3:]),np.trace(cov[:3,:3]),self.imm.noise,self.imm.nis,*self.imm.weights,np.linalg.norm(state[:3]),*angles(state[:3]),len(self.history)/100,obs.t/360]
            rows.append(gate+local+list(archive)+list(d['harmonic']))
        return np.array(rows) # 60 features, plus h/axis flags => 63 (<64 budget)

def apply_gate(package,gate,features,name,expert):
    inputs = gate.inputs(features,package) if hasattr(gate,"inputs") else features[:,:16]
    package.centers[name]=gate.predict(inputs,package.centers[expert],package.centers['I'])
    package.diagnostics.setdefault('gate_weights',{})[name]=gate.weights(inputs)
    return package
