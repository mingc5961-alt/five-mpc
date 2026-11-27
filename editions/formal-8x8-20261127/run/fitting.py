import numpy as np
import torch,pickle
from scipy.optimize import minimize
from center import *
from refine.samples import records,paths
from refine import tubes
from routeclock.validity import finite_quantile
from learning import angular
def atom_npz(path,**arrays):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
def stream(v,split):
    if v=='arch':
        actual={'fit':'train','val':'validation','shape':'train','cal':'calibration'}[split]
        for path in paths(actual):
            idx=int(path.stem)
            if split=='fit' and idx%5==0 or split=='shape' and idx%5!=0:continue
            for h,y in records(path):yield h,y
    else:
        offset=1000 if v=='half' else 2000
        ids={'fit':range(12),'val':range(12,16),'shape':range(16,20),'cal':range(20,28)}[split]
        from refine.network import interpolate
        for w in ids:
            with worldfile(offset+w).open('rb') as f:world=pickle.load(f)
            for j in range(16):
                # For calibration use one pre-registered origin per trajectory.
                origins=[180+3*((w*11+j*7)%57)] if split=='cal' else [180,216,252,288,324]
                seq=[world.observe(j,float(t)) for t in range(0,max(origins)+1,3)]
                for t in origins:yield seq[:t//3+1],interpolate(world.positions[j],t+tubes.DENSE,world.dt)
def shard(v,split):
    p=BASE/'cache'/f'{v}_{split}.npz';xs=[];ys=[];ii=[]
    for h,y in stream(v,split):
        h=window(h,180);xs.append(features(h,180));ii.append(replay_imm(h).forecast(HS)[0]);ys.append(y[::25])
    atom_npz(p,x=xs,y=ys,i=ii);return dict(rows=len(ys),path=str(p))
def train(a,v):
    dev='cuda';n=int(a[1:])
    def get(split):
        with np.load(BASE/'cache'/f'{v}_{split}.npz') as d:
            return [torch.tensor(z,device=dev,dtype=torch.float32) for z in (d['x'][:,-2*n:],d['i'],d['y'])]
    x,i,y=get('fit');vx,vi,vy=get('val');best=np.inf;traces={}
    def loss(c,y):
        d=angular(c[:,1:])-angular(y[:,1:])
        return torch.atan2(d.sin(),d.cos()).norm(dim=-1).mean()*50+.035*(c[:,1:]-y[:,1:]).norm(dim=-1).mean()
    for seed in (7,11,19):
        torch.manual_seed(seed);model=Residual(a).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=.001);trace=[];local=np.inf;stale=0
        for epoch in range(50):
            model.train()
            for ix in torch.randperm(len(x),device=dev).split(32):
                opt.zero_grad();l=loss(model(x[ix],i[ix]),y[ix]);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
            model.eval()
            with torch.no_grad():score=sum(float(loss(model(vx[k:k+32],vi[k:k+32]),vy[k:k+32]))*len(vy[k:k+32]) for k in range(0,len(vx),32))/len(vx)
            if not np.isfinite(score):raise ValueError('nonfinite validation')
            trace.append(score)
            if score<best:
                best=score;choice=[seed,epoch+1];p=modelpath(a,v);tmp=p.with_suffix('.part');torch.save(model.state_dict(),tmp);tmp.replace(p)
            if score<local-1e-5:stale=0;local=score
            else:stale+=1
            if stale>=7:break
        traces[str(seed)]=trace
    return dict(selected=choice,traces=traces,train_rows=len(x),validation_rows=len(vx),sha=sha(modelpath(a,v)))
def gate(a,v):
    n=netname(a);rows=[]
    with replay.rr.memoized_arc(),torch.inference_mode():
        for h,y in stream(v,'shape'):
            h=window(h,180);i,r=route_state(h);s=short(n,v,h,i)
            rows.append((s,r,delayed_evidence(h,n,v),y[::25]))
    s,r,e,y=map(np.stack,zip(*rows));prior=[];slope=[];fixed=[]
    for j in range(21):
        def obj(par):
            w=1/(1+np.exp(-np.clip(par[0]+(par[1] if len(par)>1 else 0)*e[:,j],-30,30)))
            c=s[:,j]+w[:,None]*(r[:,j]-s[:,j])
            return np.rad2deg(np.linalg.norm(angle_error(angles(c),angles(y[:,j])),axis=1)).mean()+.0005*np.square(par).sum()
        a0=minimize(obj,[0.],bounds=[(-8,8)]).x[0];fixed.append(1/(1+np.exp(-a0)))
        p=minimize(obj,[a0,.5],bounds=[(-8,8),(0,4)]).x;prior.append(p[0]);slope.append(p[1])
    atom_npz(BASE/'weights'/f'gate_{a}_{v}.npz',prior=prior,beta=slope,fixed=fixed)
    return dict(rows=len(y),fixed=fixed)
def calibrate(a,H,v,mode):
    model=System(a,H,v,mode);shape=np.ones((21,2));shallow=a in ('C','IMM','T8','G8','T180','G180','T8P','G8P')
    def gather(split):
        es=[];ws=[];batch=[]
        def flush():
            cs,raw=model.raw([h for h,y in batch])
            for c,w,(_,y) in zip(cs,raw,batch):es.append(tubes.errors(c,y));ws.append(w)
            batch.clear()
        for pair in stream(v,split):
            batch.append(pair)
            if len(batch)==16:flush()
        if batch:flush()
        return np.stack(es),np.stack(ws)
    with replay.rr.memoized_arc(),torch.inference_mode():
        if shallow:
            errors,_=gather('shape');shape=np.maximum(.00001,np.maximum.accumulate(np.quantile(errors[:,::25],.9,axis=0),axis=0))
        errors,raw=gather('cal')
        if shallow:raw=np.broadcast_to(shape,raw.shape)
        scores=tubes.scores(errors,raw)
        factors=np.maximum.accumulate([finite_quantile(s,.1) for s in scores.T])
        # Original selected 180-second baseline has its archived factors.
        if H==180 and v=='arch' and a in ('Cnative','VECTOR','RAFT'):
            b=read_pickle(ROOT/'tubes'/('Chronos-2' if a=='Cnative' else a)/'7/bundle.pkl');factors=b['factors']['native' if a=='Cnative' else 'fixed']
        if H==180 and a in ('Joint07','Cexternal'):
            f= replay.head_for('joint-refit')[1] if a=='Joint07' else read_pickle(replay.rr.SPECIAL/'heads/Chronos-2/common.pkl')['factors']
            factors=f
    atom_npz(tube_file(a,H,v,mode),shape=shape,factors=factors)
    return dict(calibration_n=len(scores),coverage=(scores<=factors).mean(0).tolist())
