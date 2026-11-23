"""Chapter 6 gate arms on the original one-horizon-per-anchor training sample."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .geometry import angles, angle_error
from .predictors import Gate


class HorizonGate(Gate):
    def __init__(self, mode, seed=7):
        super().__init__(seed)
        self.mode = mode
        self.recent_spacing = True

    def inputs(self, features, package):
        if self.mode != 'g1':
            return features[:, :16]
        # Freeze the forecast context for both heads. A horizon-varying slope
        # or offset would break monotonicity even with a positive slope.
        f = features[0]
        density_proxy = f[27]*360/max(1, f[26]*100)
        spacing = package.diagnostics["recent_spacing_s"] if getattr(self,"recent_spacing",False) else density_proxy
        context = np.r_[f[1:16], np.log1p(spacing)]
        return np.column_stack((features[:,0], np.tile(context,(len(features),1))))

    def importance(self, X):
        h = np.round(10*X[:,0],5)
        return np.array([self.loss_weights[k] for k in h])

    def fit(self, X, p, i, y, l2=.001):
        h = np.round(10*X[:,0],5)
        target = angles(y)
        errs = np.sum(angle_error(angles(i),target)**2,axis=1)
        bins = ((h<=1,.74401214848),((h>1)&(h<=2),.18483473318),(h>2,.07115311834))
        self.loss_weights = {}
        for mask, mass in bins:
            levels = np.unique(h[mask])
            for k in levels:
                selected = h==k
                scale2 = max(np.mean(errs[selected]), 1e-12)
                freq = 1 if self.mode=='scale' else mass/(len(levels)*selected.mean())
                self.loss_weights[k] = freq/scale2
        weights = self.importance(X)
        context = X[:,1:] if self.mode=='g1' else X
        self.mean = context.mean(axis=0)
        self.std = np.maximum(context.std(axis=0),1e-5)
        z = self.design(context)
        g = np.log1p(h)
        n = z.shape[1]
        def objective(beta):
            eta = z @ beta[:n]
            if self.mode=='g1':
                slope = z @ beta[n:]
                eta += np.logaddexp(0,slope)*g
            w = expit(eta)
            pred = i+w[:,None]*(p-i)
            err = angle_error(angles(pred),target)
            x,ypos,height = pred.T
            dx,dy,dz = (p-i).T
            r2 = np.maximum(x*x+ypos*ypos,1e-12)
            r = np.sqrt(r2)
            direction = np.column_stack(((x*dy-ypos*dx)/r2,
                            (r*dz-height*(x*dx+ypos*dy)/r)/(r2+height**2)))
            d = 2*weights*(err*direction).sum(axis=1)*w*(1-w)/len(w)
            grad = z.T @ d
            if self.mode=='g1':
                grad = np.r_[grad,z.T @ (d*g*expit(slope))]
            penalty = beta.copy()
            penalty[::n] = 0
            return np.mean(weights*(err**2).sum(axis=1))+l2*(penalty@penalty), grad+2*l2*penalty
        size = n*(2 if self.mode=='g1' else 1)
        bounds = [(None,None)]*size
        if self.mode=='g1':
            # Holding other diagnostics fixed, larger realized P errors
            # lower the route logit; larger I errors raise it (both heads).
            for offset in (0,n):
                for j in (6,7): bounds[offset+j]=(None,0)
                for j in (8,9): bounds[offset+j]=(0,None)
        result = minimize(objective,np.zeros(size),jac=True,method='L-BFGS-B',bounds=bounds,
                          options=dict(maxiter=1000,ftol=1e-12))
        if not result.success: raise RuntimeError(result.message)
        self.beta = result.x
        return self

    def design(self, context):
        return np.column_stack((np.ones(len(context)),(context-self.mean)/self.std))

    def weights(self,X):
        if self.mode!='g1':return expit(self.design(X) @ self.beta)
        z = self.design(X[:,1:])
        a,b = np.split(self.beta,2)
        return expit(z@a + np.logaddexp(0,z@b)*np.log1p(10*X[:,0]))
