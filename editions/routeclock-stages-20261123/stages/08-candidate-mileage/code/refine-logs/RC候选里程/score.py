import sys,pickle,json
import numpy as np,pandas as pd,torch
import evaluate as e
from protocol_local import OLD,OUT,log
from mileage import MODES,load_model,infer,tempo,label,old
from routeclock.geometry import angles,angle_error
from routeclock.radio import array_gain
from refine import tubes
from features import Advanced,common,rc_features
from fit_heads import predict
from matrix import head
def cases(split):
    e.setup()
    if split not in ('fresh','replicate'):
        yield from e.dataset('world' if split=='prior' else split);return
    gate=e.rc_params()['gate'];e.runtime.OUT=OUT/'cache'
    for i in (range(50,60) if split=='fresh' else range(60,70)):
        world=e.rr.get_world(i)
        with e.rr.memoized_arc():trs=e.runtime.warm_trackers(world)
        hs=e.network.warm_history(world,180,.0031)
        for j,(h,tr) in enumerate(zip(hs,trs,strict=True)):
            p=tr.snapshots[-1];e.apply_gate(p,gate,p.features,'RC','P')
            assert np.array_equal([v.position for v in h],[v.position for v in tr.history])
            y=e.network.interpolate(world.positions[j],p.origin+tubes.DENSE,world.dt)
            yield i,j,world.groups[j],h,y,p,p.features,e.pack(h,tr.memory,tr.imm,p)
def run(split):
    models={k:load_model('cpu',k) for k in MODES};oldnet=old.load('cpu')
    advanced={k:Advanced(k) for k in ['Chronos-2','VECTOR','RAFT']}
    win=json.loads((OUT/'selection.json').read_text())['selected']
    rows=[];pending=[]
    def flush(items):
        ids,js,groups,hs,ys,ps,xs,ds=zip(*items);ys=np.stack(ys)
        data={k:torch.tensor(np.stack([r[k] for r in ds])) for k in old.KEYS if k!='truth'}
        data['past']=torch.tensor(np.stack([tempo(r,h) for r,h in zip(ds,hs)]))
        targets,conf=label(dict(data,truth=torch.tensor(ys[:,::25],dtype=torch.float32)))
        dist={};regs={};centers={'RC':np.stack([r['base'] for r in ds])}
        with torch.inference_mode():
            c,w,s,a=oldnet(data,True);centers['prev']=c.numpy();dist['prev']=s[:,None,:].expand_as(targets)
            for m,net in models.items():
                c,s,w=net(data,True);centers[m]=c.numpy();dist[m]=s
        for m,net in advanced.items():
            centers[m]=np.stack([c for c,w in net.net.batch(hs)])
        z=np.stack([rc_features(p,x) for p,x in zip(ps,xs)])
        for m,k in [('prev','joint-refit'),(win,'v2-refit')]:
            hd,fac=head(k);c=centers[m];raw=predict(hd,np.stack([common(h,a) for h,a in zip(hs,c)]),z)
            err=np.stack([tubes.errors(a,y) for a,y in zip(c,ys)])
            regs[m]=tubes.scores(err,raw)<=fac
        for m,cs in centers.items():
            for i,(c,y) in enumerate(zip(cs,ys)):
                dense=np.stack([np.interp(tubes.DENSE,e.HS,col) for col in c.T],axis=1)
                diff=dense-y
                ang=np.rad2deg(np.linalg.norm(angle_error(angles(dense),angles(y)),axis=-1))
                loss=10*np.log10(1024/array_gain(angles(dense),angles(y),(32,32)))
                fails=np.flatnonzero(loss>3);hold=.02*fails[0] if len(fails) else 10.
                for k,h in enumerate([.5,1,2,3,5,8,10]):
                    j=round(h/.02);l=int(2*h)
                    tangent=(y[j]-y[j-5])/.1;speed=np.linalg.norm(tangent)
                    along=abs(diff[j]@tangent/speed) if speed>=.5 else np.nan
                    weights=conf[i,:,l];den=float(weights.sum())
                    distance=float(((dist[m][i,:,l]-targets[i,:,l]).abs()*weights).sum()/den) if m in dist and den>1e-6 else np.nan
                    rows.append((ids[i],js[i],groups[i],m,h,np.linalg.norm(diff[j]),ang[j],along,distance,hold,loss[j]<=3,
                                 float(regs[m][i,k]) if m in regs else np.nan))
    for item in cases(split):
        pending.append(item)
        if len(pending)==16:flush(pending);pending=[]
    if pending:flush(pending)
    d=pd.DataFrame(rows,columns='id anchor group arm horizon position angle along arc_error hold safe32 cover'.split())
    assert len(d)=={'replicate':160,'fresh':160,'prior':160,'test':1200,'flight':236}[split]*7*9
    d.to_csv(OUT/(split+'.csv'),index=False);log(f'{split} 所有共同历史结果已核对，共 {len(d)} 行。')
if __name__=='__main__':run(sys.argv[1])
