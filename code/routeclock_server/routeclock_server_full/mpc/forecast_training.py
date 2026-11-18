"""从本次仿真生成的独立航线训练门控、角误差管，并计算各子阵的允许复用期。"""
import hashlib
import json
import pickle
import numpy as np
from lightgbm import LGBMRegressor
from routeclock.geometry import angles,angle_error
from routeclock.predictors import Gate,apply_gate
from routeclock.radio import Radio
from .data import GRID,interpolate,inputs,link_labels,load_config
from .setup_run import atomic,save_json,load_split,log

def fuse(pkg,gate):
    return apply_gate(pkg,gate,pkg.features,"RC","P")

def fit_gate(samples):
    """沿用原平台的 16 输入 sigmoid 门控，在真实未来角误差上直接反传。"""
    X,p,i,y=zip(*[(pkg.features[:,:16],pkg.centers["P"],pkg.centers["I"],truth[::2])
                  for pkg,truth,_ in samples])
    gate=Gate(seed=7).fit(*[np.concatenate(a) for a in (X,p,i,y)])
    if not gate.fit_status["success"] or not np.isfinite(gate.beta).all():
        raise RuntimeError(f"门控优化未收敛：{gate.fit_status}")
    return gate

def gate_json(g):
    return dict(seed=g.seed,mean=g.mean.tolist(),std=g.std.tolist(),beta=g.beta.tolist(),
                optimizer=g.fit_status)

def read_gate(path):
    obj=json.loads(path.read_text(encoding="utf8"))
    gate=Gate(obj["seed"])
    gate.mean,gate.std,gate.beta=(np.asarray(obj[k]) for k in ("mean","std","beta"))
    return gate

def residuals(pkg,truth):
    """两轴半宽的监督目标都是到该时长为止的最大角误差。方位使用环形差。"""
    err=np.abs(angle_error(angles(interpolate(pkg.centers["RC"],pkg.hs)),angles(truth)))
    return np.maximum.accumulate(err,axis=0)

def tube_rows(pkg):
    """每个查询时长各生成方位/俯仰一行，使用门控的权重和既有因果诊断。"""
    f=interpolate(pkg.features,pkg.hs)
    w=np.interp(GRID,pkg.hs,pkg.diagnostics["gate_weights"]["RC"])
    base=np.column_stack([f,w,GRID/10])
    return np.column_stack([np.repeat(base,2,axis=0),np.tile(np.eye(2),(len(GRID),1))])

class AngularTube:
    def fit(self,samples):
        X=np.concatenate([tube_rows(pkg) for pkg,_,_ in samples])
        y=np.concatenate([residuals(pkg,tr).ravel() for pkg,tr,_ in samples])
        learner=LGBMRegressor(n_estimators=240,num_leaves=15,min_child_samples=20,
                               random_state=7,n_jobs=1,verbosity=-1)
        learner.fit(X,np.log(np.maximum(y,1e-6)))
        self.booster=learner.booster_
        return self

    def raw(self,pkg):
        pred=self.booster.predict(tube_rows(pkg),num_threads=1)
        return np.maximum.accumulate(np.exp(np.clip(pred,-20,3)).reshape(-1,2),axis=0)

    def calibrate(self,samples,alpha):
        # 每个独立校准网络仅一个预先选定的发布。每个分数同时覆盖两轴及全部查询时长。
        scores=np.array([np.max(residuals(pkg,truth)/self.raw(pkg)) for pkg,truth,_ in samples])
        k=int(np.ceil((len(scores)+1)*(1-alpha)))
        if k>len(scores):
            raise ValueError("独立校准航线数不足。")
        self.factor=float(np.sort(scores)[k-1])
        self.alpha=alpha
        return scores

    def predict(self,pkg):
        return self.factor*self.raw(pkg)

def sin_range(x,r):
    """正弦在整个角区间的上下界，区间含极值点时将相应端点扩展到 ±1。"""
    lo,hi=x-r,x+r
    left,right=np.sin(lo),np.sin(hi)
    def contains(phase):
        return np.ceil((lo-phase)/(2*np.pi))<=np.floor((hi-phase)/(2*np.pi))
    return (np.where(contains(-np.pi/2),-1.,np.minimum(left,right)),
            np.where(contains(np.pi/2),1.,np.maximum(left,right)))

def beam_deadlines(pkg,tube,pairs,threshold):
    """向量化复现 Radio.worst_loss 的方向余弦矩形界，对每种束宽取首次越界之前的时间。"""
    az,el=angles(interpolate(pkg.centers["RC"],pkg.hs)).T
    sa=sin_range(az,tube[:,0])
    ce=sin_range(el+np.pi/2,tube[:,1])
    se=sin_range(el,tube[:,1])
    corners=np.stack([a*b for a in sa for b in ce])
    du=np.max(np.abs(corners-np.sin(az)*np.cos(el)),axis=0)
    dv=np.max(np.abs(np.stack(se)-np.sin(el)),axis=0)
    loss=np.zeros((len(GRID),len(pairs)))
    for d,n in zip((du,dv),np.array(pairs).T):
        z=d[:,None]*n/2
        relative=(np.sinc(z)/np.sinc(d[:,None]/2))**2
        loss+=np.where(z>=1.,120.,-10*np.log10(np.maximum(relative,1e-12)))
    ok=np.logical_and.accumulate(loss<=threshold,axis=0)
    return np.where(ok,GRID[:,None],-1.).max(axis=0)

def supervised(samples,gate,radio):
    """用本轮刚学得的融合中心生成失效/速率样本，绝不沿用旧模型发布的标签。"""
    rows=[]
    for pkg,truth,shadow in samples:
        fuse(pkg,gate)
        y,r=link_labels(pkg,truth,shadow,radio)
        rows.append((inputs(pkg),y,r))
    return tuple(np.stack(a) for a in zip(*rows))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def train_models(c,out):
    from .risk import RouteRisk
    if (out/"models.json").exists():
        load_models(out)
        log(out,"训练","三个已训练组件通过指纹校验")
        return
    log(out,"门控","从独立 gate 航线拟合 16 输入融合器")
    gate_data=load_split(out,c,"gate")
    gate=fit_gate(gate_data)
    save_json(out/"gate.json",gate_json(gate))
    training=load_split(out,c,"train")
    radio=Radio(load_config())
    train=supervised(training,gate,radio)
    log(out,"误差管","拟合融合预测的方位/俯仰累计误差尺度")
    tube=AngularTube().fit(training)
    calibration=load_split(out,c,"tube_cal")
    for pkg,_,_ in calibration:
        fuse(pkg,gate)
    scores=tube.calibrate(calibration,c["tube_alpha"])
    atomic(out/"error_tube.pkl",lambda f:pickle.dump(tube,f,protocol=5))
    log(out,"风险与速率","训练失效/条件速率头，在独立 cal 网络做概率校准")
    cal=supervised(load_split(out,c,"cal"),gate,radio)
    risk=RouteRisk()
    risk.fit(train,cal)
    atomic(out/"risk.pkl",lambda f:pickle.dump(risk,f,protocol=5))
    save_json(out/"training.json",dict(gate_publications=len(gate_data),gate_optimizer=gate.fit_status,
        tube_fit_publications=len(training),tube_cal_networks=len(scores),tube_alpha=tube.alpha,
        tube_scale=tube.factor,tube_cal_scores=scores.tolist(),risk_fit_publications=len(train[0]),
        risk_cal_publications=len(cal[0])))
    save_json(out/"models.json",{name:digest(out/name) for name in ("gate.json","error_tube.pkl","risk.pkl")})

def load_models(out):
    manifest=json.loads((out/"models.json").read_text(encoding="utf8"))
    for name in ("gate.json","error_tube.pkl","risk.pkl"):
        if digest(out/name)!=manifest[name]:
            raise ValueError(f"模型与本轮训练清单不一致：{name}")
    with (out/"risk.pkl").open("rb") as f:
        risk=pickle.load(f)
    with (out/"error_tube.pkl").open("rb") as f:
        risk.tube=pickle.load(f)
    risk.gate=read_gate(out/"gate.json")
    return risk
