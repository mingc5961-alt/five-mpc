"""Release quota-dependent work when both auxiliary N01 grids finish."""
import json,sys,time
from history import ROOT,put
from native_matrix import OUT,job_key
from paired_grid import BASE,SCALES
from policy_env import initialize
def main():
    initialize()
    from closed import freeze,quota_period
    main=json.loads((BASE/'summary.json').read_bytes())
    def source(k,b,m):return OUT/'receipts'/(job_key((k,'N01',b,m))+'.json')
    for b in (4.5,9):
        if any(not source(k,b,m).exists() for m in SCALES for k in range(1400,1408)):return
    items={}
    for b in (4.5,9):
        by={m:[json.loads(source(k,b,m).read_bytes()) for k in range(1400,1408)] for m in SCALES}
        for m,rr in by.items():
            if any(r['world']!=main['physical_sha256'][str(r['world_id'])] or r['row']!='N01' or r['scale']!=m
                   or r['budget_s']!=b or abs(sum(r['result']['ledger_s'])-180)>1e-6 for r in rr):
                raise ValueError('auxiliary N01 grid')
        policy=freeze(by)
        if not policy['feasible']:raise ValueError(f'B={b} requires its registered extension grid')
        k,period=quota_period(by,policy,b)
        items[f'N01_B{b:g}']=dict(K=k,PM_ms=period,policy=policy)
    put(OUT/'budget_freeze.json',dict(accepted=64,rows=items))
if __name__=='__main__':main()
