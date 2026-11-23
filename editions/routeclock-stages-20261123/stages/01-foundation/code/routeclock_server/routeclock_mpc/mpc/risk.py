"""累计失效概率及其对各个通信束宽期望速率的影响。"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from .data import GRID, inputs

class RouteRisk:
    def __init__(self, seed=7):
        pars=dict(n_estimators=48,min_samples_leaf=16,max_features=.85,random_state=seed,n_jobs=1)
        self.classifier=ExtraTreesClassifier(**pars)
        self.rate_heads=[ExtraTreesRegressor(**pars),ExtraTreesRegressor(**pars)]
        self.calibrator=IsotonicRegression(out_of_bounds="clip",y_min=0.,y_max=1.)

    def raw(self, X):
        # 训练与在线都以“一次发布 × 完整时域”组织。累积事件随时长只能上升。
        shape=X.shape[:-1]
        proba=self.classifier.predict_proba(X.reshape(-1,X.shape[-1]))[:,1].reshape(shape)
        return np.maximum.accumulate(proba,axis=-1)

    def fit(self, train, calibration):
        X,y,rates=train
        flat=X.reshape(-1,X.shape[-1])
        target=y.ravel().astype(int)
        if len(np.unique(target))!=2:
            raise ValueError("训练航线缺少失效或未失效样本，请增加独立训练场景。")
        self.classifier.fit(flat,target)
        for status,head in enumerate(self.rate_heads):
            mask=target==status
            head.fit(flat[mask],rates.reshape(-1,rates.shape[-1])[mask])
        cx,cy,_=calibration
        self.calibrator.fit(self.raw(cx).ravel(),cy.ravel())
        self.base=y.mean(axis=0)  # 仅时长风险消融。两组条件速率模型完全共用。

    def profile(self,pkg,age_only=False):
        """全程只接收已发布包。概率将两种路线状态下的通信速率加权，直接进入 MPC。"""
        X=inputs(pkg)
        p=self.base if age_only else self.calibrator.predict(self.raw(X))
        good,bad=[head.predict(X) for head in self.rate_heads]
        rate=(1-p[:,None])*good+p[:,None]*bad
        return Profile(pkg,p,rate)

class Profile:
    def __init__(self,pkg,p,rates):
        self.pkg,self.p,self.rates=pkg,np.asarray(p),rates

    def at(self,age):
        """对每个束宽分别插值再选最优，以保证方向、宽度和收益一致。"""
        if age>GRID[-1]+1e-9:
            return 0.,0,1.
        z=np.clip(age/0.25,0,len(GRID)-1)
        j=int(z)
        nxt=min(j+1,len(GRID)-1)
        row=self.rates[j]+(z-j)*(self.rates[nxt]-self.rates[j])
        beam=int(np.argmax(row))
        return row[beam],beam,float(self.p[j]+(z-j)*(self.p[nxt]-self.p[j]))
