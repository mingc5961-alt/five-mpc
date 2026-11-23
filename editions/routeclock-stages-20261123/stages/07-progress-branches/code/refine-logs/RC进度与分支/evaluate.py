"""Common-history paired center, tube and beam evaluation. Futures are used only for scoring."""
import sys,pickle
import numpy as np,pandas as pd
from core import ROOT,log,pack,HS
from heads_new import packed_rows
from learning import MODES,load,infer
from replay_new import setup,rr,head_for
from features import Advanced,rc_features,rc_params,apply_gate,common,K
from fit_heads import predict
from refine.samples import records
from refine import tubes,network
from refine.data import ARCHIVE
from routeclock.geometry import angles,angle_error
from routeclock.validity import QUERY_H
from routeclock.radio import array_gain
from runtime import remaining_runs as runtime
def dataset(which):
    gate=rc_params()['gate']
    if which=='flight':
        from flights_local import cases
        yield from cases()
    elif which=='world':
        for w in range(40,50):
            world=rr.get_world(w)
            with rr.memoized_arc():trs=runtime.warm_trackers(world)
            histories=network.warm_history(world,180,.0031)
            for j,(h,tr) in enumerate(zip(histories,trs,strict=True)):
                p=tr.snapshots[-1];apply_gate(p,gate,p.features,'RC','P')
                assert np.array_equal([o.position for o in h],[o.position for o in tr.history])
                yield w,j,world.groups[j],h,network.interpolate(world.positions[j],p.origin+tubes.DENSE,world.dt),p,p.features,pack(h,tr.memory,tr.imm,p)
    else:
        for f in sorted((ROOT/'observations/test').glob('*.npz')):
            a=pickle.loads((ARCHIVE/'test'/(f.stem+'.pkl')).read_bytes());inputs=packed_rows(ROOT/'packed/test'/f.name,a)
            for j,((h,y),r,d) in enumerate(zip(records(f),a,inputs,strict=True)):
                yield int(f.stem),j,r['group'],h,y,r['package'],r['features']['P'],d
def main(which):
    setup(); nets={m:Advanced(m) for m in ('Chronos-2','VECTOR','RAFT')};learn={m:load('cpu',m) for m in MODES}
    heads={m:head_for(m) for m in ('Chronos-2','VECTOR','RAFT','RC','joint-transfer','joint-refit','Chronos-native')}
    from other_models import Other,ROUTE_WORK
    plus={m:Other(m) for m in ('VECTOR','RAFT')}
    routed=rr.Routed()
    for label in ('CR-special','CRG-special'):
        head=rr.HEADS[label];heads[label]=(head,head['factors'])
    for m in plus:
        head=pickle.loads((ROUTE_WORK/'heads'/m/'special.pkl').read_bytes())
        heads[m+'+P']=(head,head['factors'])
    rows=[]; pending=[]; diagnostics=[]; ceilings=[]
    def flush():
        if not pending:return
        ids,js,groups,hs,ys,ps,xs,ds=zip(*pending);ys=np.stack(ys)
        centers={'RC':np.stack([d['base'] for d in ds])};zs={'RC':np.stack([rc_features(p,x) for p,x in zip(ps,xs)])}
        for m,model in learn.items():centers[m]=infer(model,ds)
        import torch
        from learning import KEYS
        data={k:torch.tensor(np.stack([r[k] for r in ds])) for k in KEYS if k!='truth'}
        with torch.inference_mode():joint,w,dist,alpha=learn['joint'](data,details=True)
        for i,d in enumerate(ds):
            diagnostics.append((ids[i],js[i],groups[i],*w[i].tolist(),float(d['v0']),float(dist[i,10]),float(dist[i,20]),float(alpha[i,10]),float(alpha[i,20]),float(d['gate'][10]),float(d['gate'][20])))
            from ceiling import diagnostic
            for result in diagnostic(d,w[i].numpy(),dist[i].numpy(),alpha[i].numpy(),joint[i].numpy(),ys[i]):
                ceilings.append((ids[i],js[i],groups[i],*result))
        for label,sl in [('joint-no-past',slice(3,4)),('joint-no-edge',slice(1,3))]:
            ablated=[dict(d,valid=d['valid'].copy()) for d in ds]
            for row in ablated:row['valid'][sl]=0
            centers[label]=infer(learn['joint'],ablated)
        for m,net in nets.items():
            result=net.extract(hs,True);centers[m]=np.stack([c for c,z in result]);zs[m]=np.stack([z for c,z in result])
        for m,engine in plus.items():
            result=engine.online(hs,ps,True)
            centers[m+'+P']=np.stack([c for c,z in result]);zs[m+'+P']=np.stack([z for c,z in result])
        routed_rows=routed.batch(hs,ps)
        centers['CR-special']=np.stack([c for c,z in routed_rows]);zs['CR-special']=np.stack([z for c,z in routed_rows])
        crg=[];aug=[]
        for h,p,x,(c,z) in zip(hs,ps,xs,routed_rows,strict=True):
            fused,w=rr.fused_values(rr.ASSETS['gate'],rr.gate_x(p,x,h),c,p.centers['I'])
            crg.append(fused);aug.append(np.column_stack([z,w[K],c[K]-p.centers['I'][K]]))
        centers['CRG-special']=np.stack(crg);zs['CRG-special']=np.stack(aug)
        native=nets['Chronos-2'].net.batch(hs)
        np.testing.assert_allclose(centers['Chronos-2'],np.stack([c for c,w in native]),atol=1e-5)
        centers['Chronos-native']=centers['Chronos-2']
        for arm in ('joint-transfer','joint-refit'):centers[arm]=centers['joint'];zs[arm]=zs['RC']
        for m,cs in centers.items():
            error=np.stack([tubes.errors(c,y) for c,y in zip(cs,ys)])
            if m in heads:
                head,fac=heads[m]
                raw=np.stack([w for c,w in native]) if m=='Chronos-native' else predict(head,np.stack([common(h,c) for h,c in zip(hs,cs)]),zs[m])
                covered=tubes.scores(error,raw)<=fac
                widths=np.rad2deg(raw[:,K[1:]]*fac[None,:,None])
            for i,(c,y) in enumerate(zip(cs,ys)):
                dense=np.stack([np.interp(tubes.DENSE,HS,col) for col in c.T],axis=1)
                angular=np.rad2deg(np.linalg.norm(angle_error(angles(dense),angles(y)),axis=1))
                dist=np.linalg.norm(dense-y,axis=1); loss=10*np.log10(1024/array_gain(angles(dense),angles(y),(32,32)))
                fail=np.flatnonzero(loss>3); hold=.02*fail[0] if len(fail) else 10.
                for k,h in enumerate(QUERY_H[1:]):
                    j=round(h/.02)
                    rows.append((ids[i],js[i],groups[i],m,h,dist[j],angular[j],loss[j]<=3,hold,
                                 covered[i,k] if m in heads else np.nan,
                                 widths[i,k,0] if m in heads else np.nan,widths[i,k,1] if m in heads else np.nan,
                                 ds[i]['valid'][1:3].sum(),ds[i]['valid'][3],ds[i]['meta'][3,0]))
        pending.clear()
    for item in dataset(which):
        pending.append(item)
        if len(pending)==16:flush()
    flush()
    table=pd.DataFrame(rows,columns='id anchor group arm horizon position_m angle_deg safe32 hold32_s cover half_az_deg half_el_deg outgoing past_candidate match_cost'.split())
    assert table.id.nunique()=={'test':400,'world':10,'flight':59}[which] and table.arm.nunique()==17
    tmp=ROOT/(which+'.tmp.csv');table.to_csv(tmp,index=False);tmp.replace(ROOT/(which+'.csv'))
    pd.DataFrame(diagnostics,columns='id anchor group weight_current weight_edge1 weight_edge2 weight_past initial_speed distance5 distance10 effective5 effective10 old_gate5 old_gate10'.split()).to_csv(ROOT/(which+'_decoder.csv'),index=False)
    pd.DataFrame(ceilings,columns='id anchor group horizon oracle position_m angle_deg'.split()).to_csv(ROOT/(which+'_oracle.csv'),index=False)
    log(f'{which} 同历史指标已完成，{table.id.nunique()} 个独立单位，{len(table)} 行，含 7 个期限的中心、窄束保持与双角前缀覆盖。')
if __name__=='__main__':main(sys.argv[1])
