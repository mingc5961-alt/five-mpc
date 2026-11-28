"""Validate reward-window bookkeeping and fix normalizers on training inputs alone."""
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing,os,time
import numpy as np
from history import put
from policy_env import episode,initialize
from deterministic_matrix import budgets
from paired_grid import BASE
from slots import compute_slot
OUT=BASE/'learning'
def one(job):
    import torch
    from aoi_policy import Policy
    from aoi_reference import QuotaGrid
    from paired_grid import load_checked
    from paired_models import Predictor
    from closed import simulate_controls
    import paid_full as pf
    k,kind,b,q=job
    with compute_slot():
        w,_=load_checked('train',k);w.cfg['simulation']['sensing_budget']=b/180
        model=Predictor(w,kind)
        events,result,_=simulate_controls(w,'oldest',model.predict,model.factors,pf,b,.75,q)
        grid=QuotaGrid(budget_us=round(b*1e6),quota=q)
        assert len(events)==q and abs(result['ledger_s'][0]-q*.0031)<1e-7
        return b,q,sum(result['bits']),grid.opportunities
def main():
    initialize();import torch
    from aoi_policy import Policy
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(k,kind,b,q) for b,q in budgets() for k in (1000,1001) for kind in ('Acal','M')]
    with ProcessPoolExecutor(12,mp_context=multiprocessing.get_context('fork')) as pool:r=list(pool.map(one,jobs))
    normalizers={f'{b:g}_{q}':float(np.mean([bits/steps for B,Q,bits,steps in r if B==b and Q==q])) for b,q in budgets()}
    torch.manual_seed(53);net=Policy().eval();checked=[]
    for protocol in ('fixed','joint'):
        b,q=budgets()[0]
        with compute_slot():
            result,events,examples=episode('train',1002,'Acal',protocol,net,b,q,.75,np.random.default_rng(53),.2,True)
        checked.append(dict(protocol=protocol,exact_fixes=len(events),bits_error=float(examples['bits'].sum()-sum(result['bits'])),
                fail_count=int(examples['fail'].sum()),ticks=len(examples['bits'])))
    put(OUT/'ready.json',dict(bits_per_window=normalizers,normalizers_training_only=[1000,1001],checked=checked,common_multiplier=.75,
        risk_normalizer='16*9000/n_opportunities',queue='q/K before action, penalty applies only to discretionary purchases after mandatory initialization',
        reward='bits/b0 - lambda * failing_probes/f0 - eta * (queue/K) * I',tuning_seed=53,formal_seeds=[17,29,41],
        fit_worlds=list(range(1000,1064)),validation_worlds=list(range(1100,1116)),data_seen_before_lock='training inputs only'))
if __name__=='__main__':main()
