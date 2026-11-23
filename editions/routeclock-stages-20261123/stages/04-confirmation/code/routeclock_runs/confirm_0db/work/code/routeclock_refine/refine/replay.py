"""Independent event-based replay of public risk and successful bits for E01."""
import gzip
import json
import numpy as np
from routeclock.config import load_config
from routeclock.radio import Radio,array_gain
from routeclock.geometry import angles
from routeclock.world import World
from .models import HS
from .data import BASE

def replay(path):
    row=json.loads(path.read_text())
    cfg=load_config();s=row['config'];cfg['simulation']=s
    world=World(cfg,'refine_v1/'+row['phase'],row['world']);radio=Radio(cfg)
    with gzip.open(path.with_name(path.stem+'_events.json.gz'),'rt') as f:events=json.load(f)
    t0=s['warmup_s'];end=t0+s['service_s'];cost=s['observation_cost_s']
    stamps=np.array([e['t'] for e in events])
    risk=0;count=0
    for t in np.arange(round(s['service_s']/.02))*.02+t0+.01:
        event=events[np.searchsorted(stamps,t,side='right')-1]
        for uid,p in enumerate(event['states']):
            count+=1
            if p['deadline'] is None or t>=p['origin']+p['deadline']:
                risk+=1;continue
            pos=[np.interp(t-p['origin'],HS,x) for x in np.asarray(p['center']).T]
            a=angles(pos);target=angles(world.truth(uid,t)[0])
            risk+=10*np.log10(np.prod(p['pair'])/array_gain(a,target,p['pair']))>3
    np.testing.assert_allclose(risk/count,row['R'],atol=1e-12,rtol=0)
    bits=np.zeros(world.n);rr=0
    for j,event in enumerate(events):
        start=event['t'];stop=events[j+1]['t']-cost if j+1<len(events) else end
        while start<stop-1e-10:
            finish=min(start+s['tdma_quantum_s'],stop);uid=rr%world.n;rr+=1
            p=event['states'][uid];left=min(start+s['beam_switch_s'],finish);start=finish
            if p['deadline'] is None:continue
            right=min(finish,p['origin']+p['deadline'])
            if right<=left:continue
            t=(left+right)*.5
            pred=np.array([np.interp(t-p['origin'],HS,x) for x in np.array(p['center']).T])
            xyz=world.truth(uid,t)[0];a=angles(pred);pair=p['pair']
            actual=radio.snr(xyz,a,pair,world.shadow(uid,t))
            rate,_=radio.goodput(actual,radio.mcs(radio.snr(pred,a,pair)))
            bits[uid]+=rate*(right-left)
    np.testing.assert_allclose(bits,row['bits'],rtol=1e-8,atol=10.)
    return dict(model=row['model'],tube=row['tube'],world=row['world'],risk=risk/count,
                max_bit_difference=float(max(abs(bits-row['bits']))))
if __name__=='__main__':
    paths=sorted((BASE/'tube_network/dev').glob('*/*/*.json'))
    if not paths:raise ValueError('No development traces')
    results=[replay(f) for f in paths]
    (BASE/'pilot/replay.json').write_text(json.dumps(results,indent=2))
    print(results)
