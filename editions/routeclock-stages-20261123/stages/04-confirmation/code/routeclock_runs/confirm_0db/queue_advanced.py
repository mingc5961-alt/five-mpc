"""Independent, non-overlapping development and confirmation jobs."""
import fcntl
import json
import sys
import numpy as np
from advanced_tubes import Predictor,get_bundle,ROOT
from run_advanced import cells

def atomic(path,record):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,allow_nan=False));tmp.replace(path)

def main(model,stage,seed,lo,hi):
    if (ROOT/'state/frontier_verified.json').is_file():
        from parallel_frontier import frontier
        from refine import network
        network.frontier=frontier
    net=Predictor(model,seed)
    with (ROOT/'state'/f'calibration_{model}_{seed}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        bundle=get_bundle(net,seed)
    selection=ROOT/'tubes'/model/'selection.json'
    if stage=='development':
        methods=bundle['candidates']
        cells(net,seed,bundle,stage,methods,range(lo,hi))
    else:
        cells(net,seed,bundle,stage,[json.loads(selection.read_text())['tube']],range(lo,hi))

def select(model):
    methods=json.loads((ROOT/'tubes'/model/'7/offline.json').read_text())['candidates']
    rows=[json.loads(p.read_text()) for p in (ROOT/'results/development'/model/'7').glob('*/complete.json')]
    scores={}
    for m in methods:
        unique={(r['world'],r['budget']):r for r in rows if r['tube']==m}
        assert set(unique)=={(w,b) for w in range(10) for b in 'TML'}
        scores[m]=float(np.mean([r['result']['G_mbps'] for k,r in sorted(unique.items())]))
    # Paired tubes must share the same initial point prediction.
    assert all(len({r['result']['initial_center_sha256'] for r in rows if r['world']==w and r['budget']==b})==1 for w in range(10) for b in 'TML')
    atomic(ROOT/'tubes'/model/'selection.json',dict(tube=min(scores,key=lambda m:(-scores[m],m)),development_mean_mbps=scores))

if __name__=='__main__':
    if sys.argv[1]=='select':select(sys.argv[2])
    else:main(sys.argv[1],sys.argv[2],*map(int,sys.argv[3:]))
