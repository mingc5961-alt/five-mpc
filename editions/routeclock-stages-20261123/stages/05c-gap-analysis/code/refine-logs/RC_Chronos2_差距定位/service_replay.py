"""Replay a systematic subset of the tight-budget campaign for beam-specific bits and on-policy anchors."""
import sys,gzip,pickle,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from common_history import ROOT,network,world_cache,load_config,warm,MeasuredFlight,CausalModel,Predictor,get_bundle,angle_error,angles,Radio
from runtime.remaining_runs import BUDGET,controller
from runtime.fast_ledger import DeferredLedger
from parallel_frontier import frontier
from routeclock.radio import array_gain

class BeamLedger(DeferredLedger):
    def result(self,cost,rho):
        slots=np.concatenate([np.column_stack([np.full(len(u),e),u,l,r]) for e,u,l,r in self.slots])
        state=np.asarray(self.states); inputs=[]
        for uid in range(self.world.n):
            times=slots[slots[:,1]==uid]; ref=state[times[:,0].astype(int),uid]
            left,right=times[:,2:].T
            stop=np.where(ref[:,2]>0,np.maximum(left,np.minimum(right,ref[:,0]+ref[:,2])),left)
            used=stop>left
            if used.any():inputs.append(((stop-left)[used],ref[used,3:]))
        totals=np.zeros(2);calls=iter(inputs); original=self.radio.goodput
        def account(snr,mcs):
            rate,bler=original(snr,mcs);dt,pairs=next(calls);narrow=(pairs==32).all(axis=1)
            totals[0]+=rate[narrow]@dt[narrow];totals[1]+=rate[~narrow]@dt[~narrow]
            return rate,bler
        with patch.object(self.radio,'goodput',account):out=super().result(cost,rho)
        assert next(calls,None) is None
        np.testing.assert_allclose(totals.sum(),sum(out['bits']),rtol=1e-12)
        out['G32'],out['Gother']=totals/180e6
        return out

def run(shard):
    base=ROOT/'work/baseline_cache';world_cache.BASE=base
    rcpar=pickle.loads((base/'classical.pkl').read_bytes())['RC']
    cn=Predictor('Chronos-2',7);cp=get_bundle(cn,7);ns=json.loads((ROOT/'scope.json').read_text())['analysis']['confirmation_namespace']
    wins=json.loads((ROOT/'work/flight_windows.json').read_text())
    jobs=[('合成',i) for i in (0,10,20,30)]
    for route in ['FAVS_R1','FAVS_R2']:
        ids=[i for i,row in enumerate(wins) if row['route']==route]
        jobs.extend(('实飞',ids[j]) for j in (0,len(ids)//2,-1))
    beam=[];snap=[]
    for domain,i in jobs[shard::4]:
        cfg=load_config();cfg['simulation']['sensing_budget']=BUDGET['T']
        if domain=='合成':
            unit=f'{i:03}'; w=world_cache.paired_world(cfg,ns,i); warm_fn=network.warm_history
            paths=[ROOT/'results/baselines'/f'{unit}_T_RC/complete.json', ROOT/'results/confirmation/Chronos-2/7'/f'{unit}_T_native/complete.json']
        else:
            cfg['simulation'].update(primary_N=1,sensing_budget=BUDGET['T']/16)
            w=MeasuredFlight(cfg,wins[i],i);unit=wins[i]['flight_id'];warm_fn=warm
            paths=[ROOT/'results/amovfly'/model/unit/'T/complete.json' for model in ('RC','Chronos-2')]
        rc=CausalModel('RC',rcpar,cfg); buckets={}; sample=[]; radio=Radio(cfg);radio.threshold-=5
        for name,model,params,tube,path in zip(['RC','Chronos-2'],[rc,cn],[rcpar,cp],['default','native'],paths):
            original=model.batch
            def instrument(hs):
                if name=='RC' and not buckets:
                    for uid,h in enumerate(hs):buckets[id(h)]=[uid,200]
                    if domain=='合成':
                        with gzip.open(base/'warm'/f'v1_{i:03}.pkl.gz','rb') as f:trs=pickle.load(f)
                        for h,tr in zip(hs,trs,strict=True):rc.trackers[id(h)]=tr
                predictions=original(hs)
                if name=='RC':
                    for h,(center,scale) in zip(hs,predictions,strict=True):
                        uid,threshold=buckets[id(h)];origin=h[-1].t
                        if origin>=threshold and threshold<=300:
                            sample.append((uid,threshold,h.copy(),center.copy(),scale.copy()))
                            buckets[id(h)][1]+=50
                return predictions
            with patch.object(model,'batch',instrument),patch.object(network,'warm_history',warm_fn),patch.object(network,'frontier',frontier if name=='Chronos-2' and domain=='合成' else network.frontier),controller(radio,10),patch.object(network,'Ledger',BeamLedger):
                result,_=network.run_world(cfg,w,model,params,tube)
            ref=json.loads(path.read_text())['result']
            beam.append(dict(domain=domain,unit=unit,model=name,replay_G=result['G_mbps'],original_G=ref['G_mbps'],
                             replay_R=result['R'],original_R=ref['R'],G32=result['G32'],Gother=result['Gother']))
        for uid,threshold,h,rc_c,rc_s in sample:
            origin=h[-1].t
            (cc,cs),=cn.batch([h])
            for horizon in (3,5):
                truth=w.truth(uid,origin+horizon)[0];k=horizon*2
                for label,c in [('RC',rc_c),('Chronos-2',cc)]:
                    theta=np.rad2deg(np.linalg.norm(angle_error(angles(c[k]),angles(truth))))
                    ages=np.arange(horizon*50+1)*.02
                    actual=network.interpolate(w.positions[uid],origin+ages,w.dt)
                    trajectory=np.column_stack([np.interp(ages,network.HS,axis) for axis in c.T])
                    safe=np.max(10*np.log10(1024/array_gain(angles(trajectory),angles(actual),(32,32))))<=3
                    for width,s,factors in [('RC',rc_s,rcpar['factors']['default']),('Chronos-2',cs,cp['factors']['native'])]:
                        acts=frontier(radio,c,s,factors)
                        cert=any(a.pair==(32,32) and a.interval==horizon for a in acts)
                        snap.append(dict(domain=domain,unit=unit,uid=uid,threshold=threshold,origin=origin,h=horizon,center=label,width=width,
                                         err_m=np.linalg.norm(c[k]-truth),err_deg=theta,cert32=cert,safe32_prefix=safe))
        print(domain,unit,len(sample),flush=True)
    d=Path(__file__).parent
    pd.DataFrame(beam).to_csv(d/f'beam_{shard}.csv',index=False)
    pd.DataFrame(snap).to_csv(d/f'service_{shard}.csv',index=False)
if __name__=='__main__':run(int(sys.argv[1]))
