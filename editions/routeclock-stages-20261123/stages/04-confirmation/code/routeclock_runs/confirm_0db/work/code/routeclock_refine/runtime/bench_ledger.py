"""Timed A/B on exactly the same event stream; assert accounting equivalence."""
import gzip,json,time
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from refine.data import BASE
from refine.world_cache import paired_world
from routeclock.config import load_config
from routeclock.scheduler import Action
from routeclock.radio import Radio
from refine.network import Ledger
from runtime.fast_ledger import DeferredLedger

def verify(folder):
    row=json.loads((folder/'native.json').read_text())
    cfg=load_config();s=row['config'];cfg['simulation']=s
    world=paired_world(cfg,'refine_v1/dev',row['world']);radio=Radio(cfg)
    with gzip.open(folder/'native_events.json.gz','rt') as f:events=json.load(f)
    epochs=[]
    for j,e in enumerate(events):
        states=[SimpleNamespace(origin=p['origin'],center=np.asarray(p['center']),action=Action(tuple(p['pair']),p['deadline'],0.,0.) if p['pair'] else None) for p in e['states']]
        stop=events[j+1]['t'] if j+1<len(events) else s['warmup_s']+s['service_s']
        start=stop-.0031 if j+1<len(events) else stop
        epochs.extend([(e['t'],start,states,False),(start,stop,states,True)])
    timings={}
    for cls in (Ledger,DeferredLedger,DeferredLedger,Ledger):
        t=time.monotonic()
        for _ in range(5):
            ledger=cls(cfg,world,radio,s['warmup_s'],s['service_s'])
            for args in epochs:ledger.advance(*args)
            result=ledger.result(.0031,.05/3)
        timings.setdefault(cls.__name__,[]).append((time.monotonic()-t)/5)
        for k in ('R','unavailable_fraction','probes_per_user'):assert result[k]==row[k],k
        for k in ('ledger_s','narrow_good_data_s','G_mbps','bits'):
            np.testing.assert_allclose(result[k],row[k],rtol=1e-11,atol=1e-7)
    print(folder.parent.name,timings,flush=True)
for model in ['moirai','chronos']:
    verify(BASE/'tube_network/dev'/model/'000')
