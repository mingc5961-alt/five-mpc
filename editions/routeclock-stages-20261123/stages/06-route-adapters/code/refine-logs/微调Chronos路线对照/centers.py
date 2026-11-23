import csv
from replay_route import *
from routeclock.radio import array_gain
from concurrent.futures import ProcessPoolExecutor
def prepare(w):
    setup()
    world=get_world(w)
    with memoized_arc():trackers=runtime.warm_trackers(world)
    obs=network.warm_history(world,180,.0031)
    for h,tr in zip(obs,trackers,strict=True):
        assert np.array_equal([o.position for o in h],[o.position for o in tr.history])
        assert np.array_equal([o.t for o in h],[o.t for o in tr.history])
    return w
def run():
    setup()
    routed=Routed();rc=rc_params()['gate']; rows=[]; hold_rows=[]
    from routeclock.radio import array_gain
    for w in range(40):
        world=get_world(w);hist=network.warm_history(world,180,.0031)
        trackers=runtime.warm_trackers(world)
        for h,tr in zip(hist,trackers,strict=True):
            assert np.array_equal([o.position for o in h],[o.position for o in tr.history])
        pkgs=[tr.snapshots[-1] for tr in trackers]
        base=routed.engine.batch(hist); direct=routed.batch(hist,pkgs)
        for uid,(h,p,(c,_),(cr,_)) in enumerate(zip(hist,pkgs,base,direct,strict=True)):
            imm=p.centers['I']; fc,_=fused_values(ASSETS['gate'],gate_x(p,p.features,h),cr,imm)
            apply_gate(p,rc,p.features,'RC','P')
            for label,center in [('C',c),('CR',cr),('CRG',fc),('RC',p.centers['RC']),('P',p.centers['P']),('I',imm)]:
                truth=network.interpolate(world.positions[uid],p.origin+QUERY_H[1:],world.dt)
                q=center[K[1:]]
                ang=np.rad2deg(np.linalg.norm(angle_error(angles(q),angles(truth)),axis=-1))
                loss=10*np.log10(1024/array_gain(angles(q),angles(truth),(32,32)))
                dense=np.arange(501)*.02
                dense_truth=network.interpolate(world.positions[uid],p.origin+dense,world.dt)
                dense_pred=np.column_stack([np.interp(dense,HS,v) for v in center.T])
                dense_loss=10*np.log10(1024/array_gain(angles(dense_pred),angles(dense_truth),(32,32)))
                fail=np.flatnonzero(dense_loss>3)
                hold_rows.append((w,uid,label,float(dense[fail[0]]) if len(fail) else 10.,int(len(fail)==0)))
                for i,dt in enumerate(QUERY_H[1:]):
                    rows.append([w,uid,world.groups[uid],label,dt,np.linalg.norm(q[i]-truth[i]),ang[i],float(loss[i]<=3),float(p.diagnostics['support'][K[i+1]])])
        if (w+1)%10==0:log(f'同历史中心核验 {w+1}/40 个确认世界，每世界 16 架、7 个预测期限；输入逐机核对。')
    import pandas as pd
    df=pd.DataFrame(rows,columns=['world','uid','group','method','horizon_s','position_m','angle_deg','safe32','route_support'])
    df.to_csv(OUT/'中心逐机.csv',index=False)
    pd.DataFrame(hold_rows,columns=['world','uid','method','safe_hold_32_s','censored_at_10']).to_csv(OUT/'固定窄束保持逐机.csv',index=False)
if __name__=='__main__':
    if sys.argv[1]=='cache':
        with ProcessPoolExecutor(int(sys.argv[2])) as ex:
            for w in ex.map(prepare,range(40)):print(w,flush=True)
    else:run()
