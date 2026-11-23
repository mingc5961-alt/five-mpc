"""Route covariate adapters for the frozen VECTOR decoder and RAFT retrieval fusion."""
import copy
from unittest.mock import patch
from replay_route import *
from train_vector_long import features as velocity_features, HORIZONS
from train_raft_long import route_ids
from torch import nn
from features import vector_features,raft_features,DiagnosticBank
from sklearn.decomposition import PCA
from fit_heads import projection

MODELS=('VECTOR','RAFT')
def volume(label,kind):return OUT/(label+'_'+kind+'.pkl')
def torch_angle_metric(delta,origin,target):
    from train_vector_long import angles as torch_angles
    err=torch_angles((delta+origin[:,None])[:,HORIZONS])-torch_angles(target[:,HORIZONS])
    return torch.linalg.vector_norm(torch.atan2(err.sin(),err.cos()),dim=-1).mean()

class Adapter(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name
        self.proj=nn.Linear(7,256 if name=='VECTOR' else 3,bias=False)
        nn.init.zeros_(self.proj.weight)
        self.register_buffer('mean',torch.zeros(7));self.register_buffer('scale',torch.ones(7))
    def forward(self,route):
        return self.proj((route-self.mean)/self.scale)

class Other:
    def __init__(self,name,load=True,device='cuda'):
        self.name=name;self.adv=Advanced(name);self.net=self.adv.net.net
        self.net.requires_grad_(False)
        self.adapter=Adapter(name).cuda()
        if load:self.adapter.load_state_dict(torch.load(OUT/(name+'_adapter.pt'),map_location='cuda',weights_only=True))
        self.adapter.eval()
    def context(self,h):
        t=np.asarray([o.t for o in h]);p=np.stack([o.position for o in h])
        if self.name=='VECTOR':return velocity_features(t,p,t[-1]),None
        return context(t,p,t[-1])[:2]
    @torch.no_grad()
    def encoded(self,histories,exclude=None):
        xs,ms=zip(*(self.context(h) for h in histories),strict=True)
        x=torch.as_tensor(np.stack(xs),dtype=torch.float32,device='cuda')
        if self.name=='VECTOR':
            v=(x[...,:3]-self.net.mean)@self.net.whiten+self.net.mask_projection(x[...,3:])
            _,hp=self.net.gru.gru1(v[:,:-20])
            _,hidden=self.net.gru.gru1(v[:,-20:],hp)
            return hidden.transpose(0,1)
        mask=torch.as_tensor(np.stack(ms),dtype=torch.float32,device='cuda')
        ex=torch.full((len(x),),-1,device='cuda') if exclude is None else torch.as_tensor(exclude,device='cuda')
        neighbors=self.adv.net.bank.retrieve(x,mask,ex)
        own=self.net.linear_x(x.transpose(1,2))
        ext=sum(layer(y.transpose(1,2)) for layer,y in zip(self.net.retrieval_pred,neighbors,strict=True))
        return torch.cat((own,ext),dim=-1)
    def decode(self,x,r):
        proj=self.adapter(r)
        if self.name=='VECTOR':
            rep,_=self.net.gru.gru2(proj,x.transpose(0,1).contiguous())
            return ((self.net.gru.fc(rep)@self.net.invert+self.net.mean)*.5).cumsum(dim=1)
        own,ext=x.split(20,dim=-1)
        return self.net.linear_pred(torch.cat((own,ext+proj.transpose(1,2)),dim=-1)).transpose(1,2)
    @torch.inference_mode()
    def online(self,histories,pkgs,enabled=True):
        if not enabled:return self.adv.extract(histories,True)
        rr=torch.tensor(np.stack([route_inputs(p,h[-1]) for p,h in zip(pkgs,histories,strict=True)]),device='cuda',dtype=torch.float32)
        add=self.adapter(rr)
        def inject(m,args):
            if self.name=='VECTOR':return args[0]+add,*args[1:]
            own,ext=args[0].split(20,dim=-1)
            return (torch.cat((own,ext+add.transpose(1,2)),dim=-1),)
        layer=self.net.gru.gru2 if self.name=='VECTOR' else self.net.linear_pred
        handle=layer.register_forward_pre_hook(inject,prepend=True)
        try:return self.adv.extract(histories,True)
        finally:handle.remove()

def training_data(engine,split):
    pending=[];features=[];routes=[];origins=[];truth=[]
    def flush():
        if not pending:return
        ids,h,package,t=zip(*pending,strict=True)
        features.append(engine.encoded(h,ids if split=='train' else None).detach().cpu())
        routes.extend([route_inputs(p,v[-1]) for v,p in zip(h,package,strict=True)])
        origins.extend(v[-1].position for v in h);truth.extend(v[25::25] for v in t);pending.clear()
    for path in paths(split):
        uid=int(path.stem)
        if split=='train' and uid%5==0:continue
        archive=pickle.loads((ARCHIVE/split/(path.stem+'.pkl')).read_bytes())
        for (h,t),a in zip(records(path),archive,strict=True):
            pending.append((uid,h,a['package'],t))
            if len(pending)==16:flush()
    flush()
    out=[torch.cat(features).cuda(),*[torch.tensor(np.stack(z),dtype=torch.float32,device='cuda') for z in (routes,origins,truth)]]
    assert len(out[0])==(2880 if split=='train' else 1200)
    return out

def train_adapter(name):
    path=OUT/(name+'_adapter.pt')
    if path.exists():return
    torch.manual_seed(7)
    eng=Other(name,False)
    t=training_data(eng,'train');v=training_data(eng,'validation')
    eng.adapter.mean.copy_(t[1].mean((0,1)));eng.adapter.scale.copy_(t[1].std((0,1)).clamp_min(1e-5))
    with torch.no_grad():
        # Route projection is exactly zero initially. Compare to standard online inference on identical observations.
        sample=paths('validation')[0];pack=pickle.loads((ARCHIVE/'validation'/(sample.stem+'.pkl')).read_bytes())
        h=[r[0] for r in records(sample)];pk=[r['package'] for r in pack]
        before=eng.online(h,pk,False);after=eng.online(h,pk,True)
        for (a,az),(b,bz) in zip(before,after,strict=True):
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(az,bz)
    if name=='VECTOR':
        assert eng.net.gru.gru2.dropout==0
        eng.net.gru.gru2.train()  # cuDNN needs the training reserve for gradients into the route input.
    opt=torch.optim.AdamW(eng.adapter.parameters(),lr=.002,weight_decay=.001)
    def metric(d,idx):return torch_angle_metric(eng.decode(d[0][idx],d[1][idx]),d[2][idx],d[3][idx])
    def validate():
        with torch.no_grad():
            return sum(float(metric(v,ix))*len(ix) for ix in torch.arange(len(v[0]),device='cuda').split(64))/len(v[0])
    best=validate();best_state=copy.deepcopy(eng.adapter.state_dict());history=[best];stale=0
    for epoch in range(50):
        for ix in torch.randperm(len(t[0]),device='cuda').split(64):
            opt.zero_grad();loss=metric(t,ix);loss.backward();torch.nn.utils.clip_grad_norm_(eng.adapter.parameters(),1.)
            opt.step()
        score=validate();history.append(score)
        if score<best:
            best=score;best_state=copy.deepcopy(eng.adapter.state_dict());stale=0
        else:stale+=1
        if stale>=6:break
    tmp=path.with_suffix('.tmp');torch.save(best_state,tmp);tmp.replace(path)
    write(OUT/(name+'_training.json'),dict(validation_angle_rad=history,best=best,train_routes=960,validation_routes=400,trainable_params=eng.adapter.proj.weight.numel(),zero_projection_exact=True))
    log(f'{name} 路线投影冻结。零投影已复现原中心和专用管特征；同一验证角损失从 {history[0]:.5f} 到 {best:.5f} rad；RAFT 训练起点的检索剔除本路线。')

ROUTE_WORK=OUT/'route_heads'
def fit_tube(name):
    import fit_heads
    from refine.tubes import errors
    dst=ROUTE_WORK/'features'/name;dst.mkdir(parents=True,exist_ok=True)
    net=Other(name)
    for split in ('train','calibration','validation'):
        dest=dst/(split+'.npz')
        if dest.exists():continue
        pending=[];rows=[]
        def flush():
            if not pending:return
            ids,h,pk,t=zip(*pending,strict=True)
            pred=net.online(h,pk)
            for uid,hist,truth,(center,z) in zip(ids,h,t,pred,strict=True):
                rows.append((uid,common(hist,center),z,errors(center,truth)))
            pending.clear()
        for path in paths(split):
            uid=int(path.stem)
            if split=='train' and uid%5!=0:continue
            archive=pickle.loads((ARCHIVE/split/(path.stem+'.pkl')).read_bytes())
            for (h,t),a in zip(records(path),archive,strict=True):
                pending.append((uid,h,a['package'],t))
                if len(pending)==12:flush()
        flush(); cols=dict(zip(('ids','common','special','error'),map(np.stack,zip(*rows,strict=True)),strict=True))
        assert len(set(cols['ids']))==(240 if split=='train' else 400)
        tmp=dest.with_suffix('.tmp.npz');np.savez(tmp,**cols);tmp.replace(dest)
    with patch.object(fit_heads,'OUT',ROUTE_WORK),patch.object(fit_heads,'KINDS',('special',)),patch.object(fit_heads,'log',lambda msg:log(name+' 路线版 '+msg)):
        fit_heads.fit(name)
    log(f'{name} 路线中心专用管完成；冻结投影后用 240 条训练路线拟合树、独立 400 条路线校准。')

class OtherArm(Model):
    def __init__(self,name,kind,world,engine,head):
        self.arm=name;self.kind=kind;self.track={};self.net=engine;self.head=head;self.only_route=True
        if kind!='base':self.warm=runtime.warm_trackers(world)
    def batch(self,histories):
        packages=None if self.kind=='base' else self.history_packages(histories)
        rows=self.net.online(histories,packages,self.kind!='base')
        widths=predict(self.head,np.stack([common(h,c) for h,(c,_) in zip(histories,rows,strict=True)]),np.stack([z for _,z in rows]))
        return [(r[0],w) for r,w in zip(rows,widths,strict=True)]

def other_cell(name,kind,w,b,engine):
    world=get_world(w); arm=name+'-'+kind
    src=(ROUTE_WORK if kind=='refit' else SPECIAL)/'heads'/name/'special.pkl'
    head=pickle.loads(src.read_bytes())
    cfg=load_config();rho=runtime.BUDGET[b];cfg['simulation']['sensing_budget']=rho
    radio=Radio(cfg);radio.threshold-=5
    model=OtherArm(name,kind,world,engine,head)
    start=time.monotonic()
    with runtime.controller(radio,10.),patch.object(network,'Ledger',HoldLedger),memoized_arc():
        r,_=network.run_world(cfg,world,model,{'factors':{arm:head['factors']}},arm)
    assert abs(sum(r['ledger_s'])-180)<1e-6 and abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-7
    assert r['ledger_s'][0]<=rho*180+.0031001 and abs(r['ledger_s'][0]-.0031*r['updates'])<1e-6
    assert r['probes_per_user']==9000 and 0<=r['unavailable_fraction']<=r['R']<=1 and 0<r['hold_mean_s']<=10.000001
    write(OUT/'other_cells'/f'{w:03}_{b}_{arm}.json',dict(arm=arm,world=w,budget=b,namespace=NS,seed=7,mcs_margin_db=0,result=r,elapsed_s=time.monotonic()-start,head_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((OUT/(name+'_adapter.pt')).read_bytes()).hexdigest(),center_sha256=CFG['center_sha256'][name]))
    count=len(list((OUT/'other_cells').glob('*.json')))
    log(f'VECTOR/RAFT 扩展闭环 {count}/180，世界 {w}、{b}、{arm}，净吞吐 {r["G_mbps"]:.2f} Mbps、风险 {100*r["R"]:.3f}%、连续保持 {r["hold_mean_s"]:.3f} s。')

def other_worker(name):
    setup();net=Other(name)
    for w in range(10):
        for b in ('T','M','L'):
            for kind in ('base','transfer','refit'):
                key=f'{w:03}_{b}_{name}-{kind}'
                with (OUT/'locks'/(key+'.lock')).open('a') as f:
                    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if not (OUT/'other_cells'/(key+'.json')).exists():other_cell(name,kind,w,b,net)

def other_centers(name):
    import pandas as pd
    setup();net=Other(name);rows=[]; holds=[]
    dense=np.arange(501)*.02
    for w in range(40):
        world=get_world(w); hs=network.warm_history(world,180,.0031); trs=runtime.warm_trackers(world)
        pk=[tr.snapshots[-1] for tr in trs]
        for on in (False,True):
            method=name+('+P' if on else '')
            for uid,((center,_),p) in enumerate(zip(net.online(hs,pk,on),pk,strict=True)):
                truth=network.interpolate(world.positions[uid],p.origin+dense,world.dt)
                pred=np.column_stack([np.interp(dense,HS,v) for v in center.T])
                loss=10*np.log10(1024/array_gain(angles(pred),angles(truth),(32,32)))
                fail=np.flatnonzero(loss>3)
                holds.append((w,uid,method,float(dense[fail[0]]) if len(fail) else 10.,int(len(fail)==0)))
                indices=np.rint(QUERY_H[1:]*50).astype(int)
                distance=np.linalg.norm(pred[indices]-truth[indices],axis=1)
                angle=np.rad2deg(np.linalg.norm(angle_error(angles(pred[indices]),angles(truth[indices])),axis=1))
                for j,h in enumerate(QUERY_H[1:]):
                    rows.append((w,uid,world.groups[uid],method,h,distance[j],angle[j],float(loss[indices[j]]<=3),float(p.diagnostics['support'][K[j+1]])))
    pd.DataFrame(rows,columns=['world','uid','group','method','horizon_s','position_m','angle_deg','safe32','route_support']).to_csv(OUT/(name+'_中心逐机.csv'),index=False)
    pd.DataFrame(holds,columns=['world','uid','method','safe_hold_32_s','censored_at_10']).to_csv(OUT/(name+'_固定窄束保持逐机.csv'),index=False)
    log(f'{name} 路线有/无的同历史外部评分完成，40 世界 ×16 架×7 个期限；另逐 20 ms 检查 32×32 连续保持。')

if __name__=='__main__':
    from routeclock.radio import array_gain
    {'tube':fit_tube,'replay':other_worker,'centers':other_centers,'train':train_adapter}[sys.argv[1]](sys.argv[2])
