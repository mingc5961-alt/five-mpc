import os,pickle,json
import numpy as np
from center import *
from base import *
from fitting import atom_npz
from refine import network
from replay_new import rr,runtime,Radio
from unittest.mock import patch
def world(w):
    with worldfile(w).open('rb') as f:return pickle.load(f)
def case_label(w,j,t):
    from routeclock.world import GROUPS
    if w.groups[j]==GROUPS[-1]:return 'random',False
    for start,end in w.detour_windows[j]:
        if start<=t<=end:return 'detour',t-start<=2
    length=np.linalg.norm(np.diff(w.references[j],axis=0),axis=1).sum()
    dist=w.interpolate(w.progress[j],t)-w.progress[j][0]
    return ('revisit' if dist>=length else 'first'),False
def panel(a,H,v,mode,w,path):
    from panel_reuse import paired_rows, read_panel, sibling_rows
    args = dict(a=a,H=H,v=v,mode=mode,w=w)
    names = [a]
    gated = a.endswith('8P')
    if gated:
        names += [a+'-pure',a+'-route',a+'-half',a+'-fixed']
    def save(keys, errors, sources, **extra):
        atom_npz(path.with_suffix('.npz'),keys=keys,errors=errors,names=names,**extra)
        return dict(n=len(errors),angles=errors.mean(0).tolist(),names=names,
                    paid_s=960*.0031,reuse=sources)
    sibling = sibling_rows(args)
    if sibling is not None:
        keys, errors, source = sibling
        return save(keys, errors, [source])

    wd = world(w)
    paired = paired_rows(args, wd, world)
    keys = [None] * 960
    errors = [None] * 960
    need_standalone = gated and mode == 'all'
    standalone = [None] * 960 if need_standalone else []
    sources = []
    shared = []
    if paired:
        old, shared, source = paired
        old_pure = read_panel(dict(args, w=w-3000, a=netname(a))) if need_standalone else None
        if need_standalone and old_pure is None:
            shared = []
        if shared:
            sources.append(source)
            if old_pure:
                sources.append(dict(old_pure[1], aircraft=shared, origins=60*len(shared)))
            for tick in range(60):
                for j in shared:
                    off = 16*tick+j
                    keys[off], errors[off] = old['keys'][off], old['errors'][off]
                    if old_pure:
                        standalone[off] = old_pure[0]['errors'][off]
    fresh = [j for j in range(16) if j not in shared]
    hs = network.warm_history(wd,180,.0031)
    model = System(a,H,v,mode)
    if gated:
        with np.load(BASE/'weights'/f'gate_{a}_{v}.npz') as z: fixed=z['fixed']
    def score(centers, truth):
        return np.rad2deg(np.linalg.norm(angle_error(angles(np.stack(centers)[:,[2,4,10,20]]),angles(truth)),axis=-1))
    with rr.memoized_arc(),torch.inference_mode():
        for tick in range(60):
            for j in fresh:
                t=180+3*tick+3*j/16+.0031
                hs[j].append(wd.observe(j,t))
            if not fresh:
                break
            centers,_ = model.raw([hs[j] for j in fresh])
            for index,(j,c) in enumerate(zip(fresh,centers,strict=True)):
                t = hs[j][-1].t
                label,early=case_label(wd,j,t)
                truth=network.interpolate(wd.positions[j],t+np.array([1,2,5,10]),wd.dt)
                variants=[c]
                off=16*tick+j
                if gated:
                    short,route,weight=model.aux[index]
                    variants.extend([short,route,(short+route)/2,short+fixed[:,None]*(route-short)])
                    if need_standalone:
                        # The standalone predictor scores the original float32 center.
                        standalone[off]=score([short],truth)
                errors[off]=score(variants,truth)
                keys[off]=(j,t,label,early,wd.groups[j])
    extra = {'standalone_errors':np.stack(standalone)} if need_standalone else {}
    return save(np.array(keys),np.stack(errors),sources,**extra)
def cell(a,H,v,mode,w,b,m,transfer,inference_device="cpu"):
    cfg=load_config();cfg['simulation']['sensing_budget']=rho=runtime.BUDGET[b];radio=Radio(cfg);radio.threshold-=5
    obj=System(a,H,v,mode,inference_device);obj.setup_tube(transfer);wd=world(w)
    sums=np.zeros(4);count=0
    class Scorer:
        def batch(self,hs):
            nonlocal count
            if not hasattr(self,'ids'):self.ids={id(h):j for j,h in enumerate(hs)}
            res=obj.batch(hs)
            for h,(c,_) in zip(hs,res):
                y=network.interpolate(wd.positions[self.ids[id(h)]],h[-1].t+np.array([1,2,5,10]),wd.dt)
                sums[:]+=np.rad2deg(np.linalg.norm(angle_error(angles(c[[2,4,10,20]]),angles(y)),axis=1));count+=1
            return res
    rr.setup()
    with rr.memoized_arc(),torch.inference_mode(),runtime.controller(radio,10.),patch.object(network,'Ledger',rr.HoldLedger):
        r,_=network.run_world(cfg,wd,Scorer(),{'factors':{a:np.asarray(obj.factors)*m}},a)
    assert abs(sum(r['ledger_s'])-180)<1e-6 and abs(r['ledger_s'][0]-.0031*r['updates'])<1e-6 and r['ledger_s'][0]<=rho*180+.0031001
    assert r['probes_per_user']==9000 and abs(sum(r['bits'])/180e6-r['G_mbps'])<1e-6 and 0<=r['unavailable_fraction']<=r['R']<=1
    return dict(result=r,origins=count,closed_loop_deg=(sums/count).tolist())
