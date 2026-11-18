import time
import numpy as np
from .predictors import Tracker,apply_gate
from .radio import Radio
from .geometry import angles
from .scheduler import frontier,choose_actions,TokenBucket,Action
from .config import dump_json

def simulate(cfg,world,bundle,method='RC',chronos=None,out=None,control=None,scheduler='value',action_mode='joint',service_trackers=None,compact=False):
    chronos_center={'C':'C','CR':'CR_raw','CR_raw':'CR_raw'}.get(method,False)
    if chronos_center and chronos is None:raise RuntimeError('Actual Chronos model required')
    if method=='F' and bundle.get('fixed') is None:raise RuntimeError('Fixed baseline requires validation-selected configuration')
    n=world.n; sim=cfg['simulation'];warm=sim['warmup_s'];end=warm+sim['service_s'];c=sim['observation_cost_s'];rho=sim['sensing_budget']
    radio=Radio(cfg);bucket=TokenBucket(c,rho);trackers=[Tracker(cfg,chronos,no_archive=(control=='no_archive')) for _ in range(n)]
    packages=[None]*n;fronts=[[] for _ in range(n)];chosen=[None]*n;due=np.arange(n)*c/rho
    warm_interval=max(1,n*c/rho);t=0.;rr=0;bits=np.zeros(n);sensed=np.zeros(n);lost=np.zeros(n);assigned=np.zeros(n);outage=np.zeros(n)
    service_sensing=0.;service_switch=0.;service_data=0.;events=[];prediction_log=[];latencies=[];total_sensing=0.;overdue=0
    center='I' if method=='F' else ('RC' if method=='Z' else method)
    choice_trace=[]
    def prepare_center(tracker,pkg):
        expert='CR_raw' if center in ('CR','CR_raw') else 'P'
        features=tracker.features(pkg,expert)
        if center in ('RC','CR'):
            gate=bundle['controls'][center]['no_archive_gate'] if control=='no_archive' else bundle['gates'][center]
            apply_gate(pkg,gate,features,center,expert)
        return features
    def log_service_choices(at):
        choice_trace.append({'t':at,'choices':[
            {'origin':pkg.origin,'pair':act.pair if act else None,'interval':act.interval if act else 0}
            for pkg,act in zip(packages,chosen)]})
    if service_trackers is not None:
        if len(service_trackers)!=n or control is not None or scheduler!='value':
            raise ValueError('Prewarmed evaluation expects one tracker per vehicle and the common value scheduler')
        trackers=service_trackers
        t=warm
        for k,tracker in enumerate(trackers):
            pkg=tracker.snapshots[-1]
            if pkg.origin>warm or warm-pkg.origin>pkg.hs[-1]:
                raise ValueError('Prewarmed forecast unavailable at service start')
            packages[k]=pkg
            tracker.chronos=chronos
            f=prepare_center(tracker,pkg)
            if method=='F' or action_mode=='fixed':
                fixed=bundle['fixed'];env=None
                fronts[k]=[Action(tuple(fixed['pair']),float(fixed['interval']),0.,0.)]
            else:
                env=bundle['heads'][center].predict(pkg,f)
                fronts[k]=frontier(pkg,center,env,radio,cfg)
            prediction_log.append({'uav':k,'origin':pkg.origin,'h':pkg.hs,'center':pkg.centers[center],
                                   'gate_weights':pkg.diagnostics.get('gate_weights',{}),'envelope_rad':env,'frontier':[a.__dict__ for a in fronts[k]]})
        reference=bundle.get('action_reference',{'pair':(16,16),'interval':1.})
        chosen=choose_actions(fronts,c,rho,scheduler,action_mode=='width_only',action_mode=='interval_only',reference['interval'],tuple(reference['pair']))
        due=np.array([pkg.origin+(act.interval if act else 0.) for pkg,act in zip(packages,chosen)])
        log_service_choices(t)
    def transfer(start,finish,i):
        nonlocal service_data
        if finish<=start:return {'delivered_bits':0.}
        left=max(start,warm);right=min(finish,end)
        if right<=left:return {'delivered_bits':0.,'phase':'warmup'}
        duration=right-left;service_data+=duration;assigned[i]+=duration
        package=packages[i];act=chosen[i];mid=(left+right)/2
        if package is None or act is None or mid-package.origin>package.hs[-1]+1e-8:
            lost[i]+=duration;return {'delivered_bits':0.,'unavailable':True,'data_s':duration}
        p=package.at(center,mid-package.origin);pointing=angles(p);truth,_=world.truth(i,mid)
        expected=radio.snr(p,pointing,act.pair);snr=radio.snr(truth,pointing,act.pair,world.shadow(i,mid))
        rate,bler=radio.goodput(snr,radio.mcs(expected));bits[i]+=duration*rate
        if snr<radio.threshold[0]:outage[i]+=duration
        from .geometry import angle_error
        return {'delivered_bits':float(duration*rate),'data_s':duration,'unavailable':False,'snr_db':float(snr),'predicted_snr_db':float(expected),'mcs':radio.mcs(expected),'bler':float(bler),'pointing':pointing,'actual_angle_error':angle_error(angles(truth),pointing),'pair':act.pair,'prediction_origin':package.origin,'prediction_age':mid-package.origin,'deadline':package.origin+act.interval}
    while t<end-1e-10:
        i=int(np.argmin(due)); earliest=max(float(due[i]),bucket.ready(t))
        if earliest<=t+1e-9:
            start=t;finish=min(t+c,end)
            if finish-start<c-1e-10:
                detail=transfer(start,finish,rr%n)
                events.append({'type':'data','uav':rr%n,'start':start,'finish':finish,'switch':0.,**detail});t=finish;break
            bucket.consume(t); total_sensing+=c
            if out is not None and not choice_trace and start<=warm<=finish and all(pkg is not None for pkg in packages):
                log_service_choices(warm)
            service_overlap=max(0,finish-max(t,warm));service_sensing+=service_overlap;sensed[i]+=service_overlap
            overdue+=int(t>due[i]+.01 and t>=warm)
            t=finish; obs=world.observe(i,t);start_cpu=time.perf_counter();trackers[i].observe(obs)
            pkg=trackers[i].issue(include_chronos=chronos_center)
            f=prepare_center(trackers[i],pkg)
            packages[i]=pkg
            env=None
            if method=='F' or action_mode=='fixed':
                fixed=bundle['fixed'];chosen[i]=Action(tuple(fixed['pair']),float(fixed['interval']),0.,0.);fronts[i]=[chosen[i]]
            else:
                head=bundle['heads'][center];action_head=None
                if control in ['global','local','no_archive']:head=bundle['controls'][center][control]
                if control=='duration' or method=='Z':action_head=bundle['controls'][center]['duration' if control=='duration' else 'snr']
                env=head.predict(pkg,f);fronts[i]=frontier(pkg,center,env,radio,cfg,action_head,f)
                reference=bundle.get('action_reference',{'pair':(16,16),'interval':1.})
                chosen=choose_actions(fronts,c,rho,scheduler,action_mode=='width_only',action_mode=='interval_only',reference['interval'],tuple(reference['pair']))
            for k in range(n):
                if packages[k] is None:continue
                if t<warm:due[k]=packages[k].origin+warm_interval
                else:due[k]=packages[k].origin+(chosen[k].interval if chosen[k] else 0)
            if out is not None and t>=warm:
                log_service_choices(t)
            latencies.append(time.perf_counter()-start_cpu)
            events.append({'type':'observation','uav':i,'start':start,'finish':finish,'cost':c,'origin':pkg.origin,'next_due':float(due[i]),'pair':chosen[i].pair if chosen[i] else None,'interval':chosen[i].interval if chosen[i] else 0,'memory':pkg.diagnostics['memory']})
            prediction_log.append({'uav':i,'origin':pkg.origin,'h':pkg.hs,'center':pkg.centers[center],'experts':pkg.centers,'envelope_rad':env,'features':f,'support':pkg.diagnostics['support'],'progress_std':pkg.diagnostics['progress_std'],'modes':pkg.diagnostics['modes'],'realized_evidence':pkg.diagnostics['archive'],'frontier':[a.__dict__ for a in fronts[i]]})
            prediction_log[-1]['gate_weights']=pkg.diagnostics.get('gate_weights',{})
        else:
            finish=min(t+sim['tdma_quantum_s'],earliest,end)
            if out is not None and not choice_trace and t<=warm<=finish and all(pkg is not None for pkg in packages):
                log_service_choices(warm)
            switch=min(sim['beam_switch_s'],finish-t);service_switch+=max(0,min(t+switch,end)-max(t,warm))
            detail=transfer(t+switch,finish,rr%n);events.append({'type':'data','uav':rr%n,'start':t,'finish':finish,'switch':switch,**detail});rr+=1;t=finish
    service=sim['service_s'];accounted=service_sensing+service_switch+service_data
    if abs(accounted-service)>1e-6:raise AssertionError(f'Time ledger {accounted} != {service}')
    if total_sensing>rho*end+c+1e-8:raise AssertionError('Global budget violated')
    result={'method':method,'head':bundle['kind'],'task':f'{world.namespace}/{world.mission}','N':n,'seed':bundle['seed'],'control':control,'scheduler':scheduler,'action_mode':action_mode,'net_mbps':float(bits.sum()/service/1e6),'uav_mbps':bits/service/1e6,'worst10_mbps':float(np.mean(np.sort(bits/service/1e6)[:max(1,int(np.ceil(n*.1)))])),'sensing_s':service_sensing,'sensing_fraction':service_sensing/service,'switch_s':service_switch,'data_s':service_data,'accounted_s':accounted,'unavailable_s':float(lost.sum()),'outage_data_fraction':float(outage.sum()/max(assigned.sum(),1e-12)),'unavailable_service_fraction':float(lost.sum()/service),'observations_total':len(latencies),'observations_service':sum(e['type']=='observation' and e['finish']>warm for e in events),'overdue_requests':overdue,'mean_update_cpu_s':float(np.mean(latencies)),'max_update_cpu_s':float(max(latencies)),'group_resources':[{'group':g,'bits':float(bits[k]),'sensing_s':float(sensed[k]),'assigned_s':float(assigned[k]),'unavailable_s':float(lost[k])} for k,g in enumerate(world.groups)]}
    result.update({'sensing_budget':rho,'warmup_s':warm,'service_s':service,'structurally_reusable_fraction':sum(g!='no_reusable_path' for g in world.groups)/n})
    if out is not None:
        from pathlib import Path
        from .holding import score_holds
        if choice_trace and abs(choice_trace[0]['t']-warm)<1e-7:
            result.update(score_holds(world,prediction_log,choice_trace,warm,end))
        out=Path(out);dump_json(out/'metrics.json',result)
        if compact:
            if center in ('RC','CR'):
                np.savez_compressed(out/'gate_weights.npz',uid=[p['uav'] for p in prediction_log],
                        origin=[p['origin'] for p in prediction_log],h=prediction_log[0]['h'],
                        weights=[p['gate_weights'][center] for p in prediction_log])
        else:
            dump_json(out/'events.json',events);dump_json(out/'predictions.json',prediction_log)
            dump_json(out/'choices.json',choice_trace)
    return result
