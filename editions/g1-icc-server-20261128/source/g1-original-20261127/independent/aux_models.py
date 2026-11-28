"""Train retrieval/snapshot models on the locked bank and calibrate RAFT/IMM without confirmation scores."""
import json,os,pickle,sys
from contextlib import ExitStack
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from history import ROOT,put
from physical_inputs import SPLITS,paths,sha_file
from slots import compute_slot
BASE=ROOT/'paired_linux_v2/auxiliary'
os.environ['G1_RUN']=str(BASE)
SCI=ROOT.parent/'g1-server-handoff/science'
sys.path.insert(0,str(SCI))

def calibrate():
    import numpy as np
    from lightgbm import LGBMRegressor
    from independent_tubes import one,H
    with ProcessPoolExecutor(6) as pool:
        fit=list(pool.map(one,[('tube_fit',k) for k in SPLITS['tube_fit']]))
        x,y=[np.concatenate(c) for c in zip(*fit)]
        tree=LGBMRegressor(objective='quantile',alpha=.90,n_estimators=110,num_leaves=12,min_child_samples=85,
                            learning_rate=.055,n_jobs=1,verbosity=-1,random_state=7).fit(x,y).booster_
        dest=BASE/'models/raft_width.pkl';tmp=dest.with_suffix('.tmp')
        with tmp.open('wb') as fp:pickle.dump(tree,fp)
        tmp.replace(dest)
        rows=list(pool.map(one,[('tube_calibration',k) for k in SPLITS['tube_calibration']]))
    rank=int(np.ceil(.9*(len(rows)+1)))
    factors={kind:np.maximum.accumulate(np.sort([r[kind] for r in rows],axis=0)[rank-1]).tolist() for kind in ('RAFT','IMM')}
    put(BASE/'models/calibration.json',dict(horizons=H.tolist(),rank=rank,fit_rows=len(x),anchors=rows,factors=factors,
        dense_truth_step_s=.02,no_confirmation_values=True))
def main():
    from raft_rebuild import train
    from snapshot_panel import train as train_snapshot
    for sub in ['worlds','models']:(BASE/sub).mkdir(parents=True,exist_ok=True)
    for s in ('train','train_validation','tube_fit','tube_calibration'):
        for k in SPLITS[s]:
            src,meta=paths(s,k);dst=BASE/'worlds'/f'{s}_{k}.pkl'
            if not dst.exists():dst.symlink_to(src)
            if dst.resolve()!=src.resolve() or sha_file(src)!=json.loads(meta.read_bytes())['file_sha256']:raise ValueError(src)
    for marker,count,fn in [('raft_fit.json',3,train),('calibration.json',6,calibrate),('snapshot_fit.json',8,train_snapshot)]:
        if (BASE/'models'/marker).is_file():continue
        with ExitStack() as leases:
            for _ in range(count):leases.enter_context(compute_slot())
            fn()
    put(BASE/'summary.json',dict(training_worlds=64,validation_worlds=16,tube_fit_worlds=16,calibration_worlds=16,
        artifacts={p.name:sha_file(p) for p in (BASE/'models').iterdir() if p.is_file()}))
if __name__=='__main__':main()
