"""Only paid observations may enter this per-aircraft memory."""
from base849 import *
from prefix849 import EMPTY,obsbytes
from functools import lru_cache
import pickle
from tube import raw
HT=np.asarray(pf.network.QUERY_H[1:])
ARMS=('C-native','GEO','C-cal','TIME','ROUTE','SHUFFLE')
LEARN=ARMS[2:]
def obsfull(o):
    return obsbytes(o)+np.asarray([o.radial_velocity,o.radial_std,*o.raw],dtype='<f8').tobytes()
def digest(prev,o):return sha(bytes.fromhex(prev)+obsfull(o))
def interp(c,d):return np.stack([np.interp(d,pf.GRID,v) for v in c.T],axis=-1)
def direction(v):return v/max(np.linalg.norm(v),1e-9)
def segment(h):
    pp=np.stack([o.position for o in h]);dt=np.diff([o.t for o in h])
    vel=np.diff(pp,axis=0)/dt[:,None]
    return (pp[-1],vel,direction(pp[-1]-pp[0]),pp-pp[-1])
class Memory:
    def __init__(self):
        self.h=[];self.pending=[];self.records=[];self.key=EMPTY
    def on_paid_fix(self,o):
        if self.h and o.t<=self.h[-1].t:raise ValueError('nonmonotone paid fix')
        for t,c,seg in self.pending:
            d=o.t-t
            if d<=10. and seg is not None:
                err=pf.difference(pf.angle(interp(c,d)),pf.angle(o.position))
                self.records.append((t,o.t,seg,d,err,np.trace(o.covariance)))
        self.pending=[v for v in self.pending if o.t-v[0]<10.]
        self.h.append(o);self.key=digest(self.key,o)
    def issue(self,c):
        seg=segment(self.h[-3:]) if len(self.h)>=3 else None
        self.pending.append((self.h[-1].t,c,seg))
    def query(self):
        z=np.zeros(10);empty=(z,z,z,0,0,0)
        if len(self.h)<3:return empty
        first=self.h[-3].t;pool=[r for r in self.records if r[0]<first and r[1]<first]
        if not pool:return empty
        p,v,heading,shape=segment(self.h[-3:])
        seg=[r[2] for r in pool]
        td=np.sqrt(np.mean((np.asarray([s[1] for s in seg])-v)**2,axis=(1,2)))
        dist=np.linalg.norm(np.asarray([s[0] for s in seg])-p,axis=1)
        cosine=np.clip(np.asarray([s[2] for s in seg])@heading,-1,1)
        shap=np.sqrt(np.mean((np.asarray([s[3] for s in seg])-shape)**2,axis=(1,2)))
        sd=td/5+dist/25+(1-cosine)*2+shap/15
        eligible=np.flatnonzero((dist<=50)&(cosine>=.5)&(shap<=30))
        def summary(keys,ids,metric,spatial):
            residual=np.asarray([pool[i][4] for i in ids]);ab=np.abs(residual)
            return np.r_[np.mean(ab,0),np.max(ab,0),np.mean([pool[i][3] for i in ids]),
                         np.mean([pool[i][5] for i in ids]),metric[keys].mean(),
                         (1-cosine[keys]).mean() if spatial else 0.,len(keys),len(pool)]
        ti=np.argsort(td,kind='stable')[:3];tf=summary(ti,ti,td,False)
        if not len(eligible):return tf,z,z,len(ti),0,0
        ri=eligible[np.argsort(sd[eligible],kind='stable')[:3]]
        shift=(int(self.key[:12],16)%(len(pool)-1)+1) if len(pool)>1 else 0
        return tf,summary(ri,ri,sd,True),summary(ri,(ri+shift)%len(pool),sd,True),len(ti),len(ri),int(shift!=0)
def features(arm,width,o,n,gap,q):
    reps=42;axis=np.tile([0.,1.],21);h=np.repeat(pf.GRID,2)
    cols=[np.log(np.maximum(width.ravel(),1e-10)),h,axis]
    shared=np.r_[o.position,np.linalg.norm(o.position),np.log1p(n),gap,o.covariance[np.triu_indices(3)]]
    time,route,shuffle,*_=q
    add=np.r_[shared,time if arm!='C-cal' else np.zeros(10),route if arm=='ROUTE' else shuffle if arm=='SHUFFLE' else np.zeros(10)]
    return np.column_stack([*cols,np.broadcast_to(add,(reps,len(add)))])
def admitted(radio,c,w,f,m):
    env=np.maximum(w,np.vstack((w[:1],w[:-1])))
    loss=radio.worst_losses(np.tile(pf.angle(c),(len(HT),1)),(m*f[:,None,None]*env).reshape(-1,2),(32,32)).reshape(len(HT),-1)
    ok=np.logical_and.accumulate(np.where(pf.GRID[None,:]<=HT[:,None],loss,0.).max(1)<=3)
    return float(max([0.,*HT[ok]]))
def geo(c,h,fallback):
    if len(h)<2:return fallback
    from routeclock.radio import array_gain
    a,b=h[-2:];vel=(b.position-a.position)/(b.t-a.t);d=np.arange(501)*.02
    loss=10*np.log10(1024/array_gain(pf.angle(interp(c,d)),pf.angle(b.position+d[:,None]*vel),(32,32)))
    ix=np.flatnonzero(loss>3);end=max(0,(ix[0]-1)*.02) if len(ix) else 10.
    return float(max([0.,*HT[HT<=end+1e-10]]))
@lru_cache
def bundle(arm):
    with (OUT/f'E1/calibration/{arm}.pkl').open('rb') as f:return pickle.load(f)
def inherited(arm,q):
    if arm in ('TIME','ROUTE','SHUFFLE') and not q[3]:return 'C-cal'
    if arm in ('ROUTE','SHUFFLE') and not q[4]:return 'TIME'
    return arm
def issued(arm,mem,width,q,smoke):
    effective=inherited(arm,q);h=mem.h;o=h[-1]
    if smoke:
        # Deterministic initial integration fixture; never used for trained endpoints.
        factor=1.+.04*(effective=='TIME')+.08*(effective in ('ROUTE','SHUFFLE'))
        return width*factor,np.asarray(get_native()),effective
    model=bundle(effective)
    x=features(effective,width,o,len(h),o.t-h[-2].t if len(h)>1 else 0.,q)
    return raw(model['tree'],x[None])[0],model['final'],effective
@lru_cache
def get_native():return np.asarray(get(OUT/'manifest.json')['factors'])
