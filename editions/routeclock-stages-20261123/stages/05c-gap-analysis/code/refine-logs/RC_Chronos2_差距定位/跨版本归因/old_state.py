"""Start the modern executor from the 19 November trackers, weights and archived observations."""
import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
from compare import OUT,CACHE,network,world_cache,load_config,Radio,ExpiryLedger,write
from routeclock.foundation import ChronosAdapter
from routeclock.predictors import apply_gate
from runtime.remaining_models import ARCHIVED, INDEX
from parallel_frontier import frontier
archive=Path('/data/chen/routeclock_archives/paired600')
sys.path.insert(0,str(archive/'tools'))
from ch6_cache import CACHE as OLD_CACHE,version,file_hash
DIR=Path('/data/chen/routeclock_runs/fixed_history_20261119/paired/cost3.1_budget1.6666667')

def run(method,w):
    folder=DIR/f'fixed_frozen_deadline_value_{method}_{w:02}'
    args=json.loads((folder/'arguments.json').read_text())
    cfg=load_config();cfg['simulation'].update(sensing_budget=args['budget_pct']/100,observation_cost_s=args['cost_ms']/1000)
    source=json.loads((archive/'configs/research.json').read_text(encoding='utf-8-sig'))
    assert source['radio']==cfg['radio']
    stripped=dict(source['simulation']);del stripped['sensing_budget'],stripped['observation_cost_s']
    key=['chapter6/test',w,{**source,'simulation':stripped}]
    cp=None;adapter=None
    if method=='C':
        ck=Path(args['checkpoint']);cp=[str(ck),[(f.name,file_hash(f)) for f in sorted(ck.iterdir()) if f.is_file()],False]
        adapter=ChronosAdapter('cuda',str(ck))
    warmkey=[key,.0031,.05,'C' if cp else False,cp]
    def load(kind,key):
        token=hashlib.sha256(json.dumps([version(),key],sort_keys=True).encode()).hexdigest()
        with gzip.open(OLD_CACHE/f'{kind}-{token}.pkl.gz','rb') as f:return pickle.load(f)
    trackers=load('warm',warmkey)
    world_cache.BASE=CACHE;world=world_cache.paired_world(cfg,'chapter6/test',w)
    ref=load('world',key)
    regen_diff={k:float(np.max(np.abs(np.asarray(getattr(ref,k))-np.asarray(getattr(world,k))))) for k in world_cache.ARRAYS}
    assert max(regen_diff.values())<1e-8 and ref.groups==world.groups
    world=ref
    world.cfg=cfg
    hsh=hashlib.sha256()
    for tr in trackers:
        tr.cfg=cfg;tr.chronos=adapter
        hsh.update(np.asarray([len(tr.history)],dtype='<i4').tobytes())
        for o in tr.history:
            for v in (o.t,o.raw,o.position,o.covariance,o.radial_velocity,o.radial_std):hsh.update(np.asarray(v,dtype='<f8').tobytes())
    assert hsh.hexdigest()==json.loads((folder/'warm_history.json').read_text())['sha256']
    bundle=pickle.loads(ARCHIVED.read_bytes());head=bundle['heads'][method]
    b={'factors':{'default':np.array([head.deadline_factors[float(h)] for h in network.QUERY_H[1:]])}}
    histories=[list(tr.history) for tr in trackers];ctx={id(h):tr for h,tr in zip(histories,trackers)}
    class Center:
        def batch(self,hs):
            out=[]
            for h in hs:
                tr=ctx[id(h)]
                for obs in h[len(tr.history):]:
                    tr.observe(obs);tr.issue(include_chronos='C' if method=='C' else False)
                p=tr.snapshots[-1];f=tr.features(p,'P')
                if method=='RC':apply_gate(p,bundle['gates']['RC'],f,'RC','P')
                out.append((p.centers[method],head.raw(p,f)[INDEX]))
            return out
    with patch.object(network,'warm_history',lambda *args:histories), patch.object(network,'Ledger',ExpiryLedger):
        r,_=network.run_world(cfg,world,Center(),b,'default')
    assert abs(sum(r['ledger_s'])-180)<1e-6 and r['probes_per_user']==9000
    dest=OUT/'exact_old';dest.mkdir(exist_ok=True)
    orig=json.loads((folder/'metrics.json').read_text())['net_mbps']
    write(dest/f'{w:02}_{method}.json',dict(world=w,method=method,archive_G=orig,modern_G=r['G_mbps'],modern_relaxed_G=r['ten_second_G'],risk=r['R'],history_hash=hsh.hexdigest(),physical_arrays_match=True,regenerated_max_abs=regen_diff,physical_source="original chapter6 cached world"))

if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
