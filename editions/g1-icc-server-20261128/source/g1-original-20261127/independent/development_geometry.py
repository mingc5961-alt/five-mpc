"""Measure the fixed common-prefix panel on development worlds while policy training runs."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import time

from geometry_panel import OUT,one
from history import put
from policy_env import initialize

def main():
    initialize()
    folder=OUT/'development'
    folder.mkdir(parents=True,exist_ok=True)
    start=time.time()
    batch=dict(start=start,total=8,workers=8,pid=os.getpid())
    put(folder/'batch.json',dict(**batch,phase='development'))
    with ProcessPoolExecutor(8) as workers:
        rows=list(workers.map(one,[('development',k) for k in range(1400,1408)]))
    if {r['world'] for r in rows}!=set(range(1400,1408)) or len({r['physical_sha256'] for r in rows})!=8:
        raise ValueError('development world coverage')
    put(folder/'summary.json',dict(worlds=8,queries=sum(r['queries'] for r in rows),
        prefixes=sum(r['prefixes'] for r in rows),closed_loop_endpoints=0,confirmation_scores_read=False))
    put(folder/'batch.json',dict(**batch,phase='done'))

if __name__=='__main__':
    main()
