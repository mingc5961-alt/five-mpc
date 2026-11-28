"""Exercise the published native event loop on fresh receipts in a separate directory."""
import argparse
import json
import os
import time
from unittest.mock import patch
from server_paths import RUN,UPSTREAM
import paid_full
import grurun
from worlds import load_world


def run(split,k,arm,m):
    target=RUN/'native'/f'{split}_{k}_{arm}_{m:.2f}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        return json.loads(target.read_bytes())
    dest=target.with_suffix(f'.{os.getpid()}.part')
    started=time.monotonic()
    with patch.object(grurun,'path',lambda *args:dest),patch.object(grurun,'world',load_world):
        grurun.run((split,k,arm,m))
    receipt=json.loads(dest.read_bytes())
    receipt.update(protocol='linux_refit_v1',elapsed_s=time.monotonic()-started,
                   model_source='pinned_Chronos_seed7' if arm=='C' else 'linux_train_split_refit')
    if split=='historical':
        ref=json.loads((UPSTREAM/'factorial_run'/f'run_confirm_{k}_{arm}_{m:.2f}.json').read_bytes())
        mismatches=[i for i,(a,b) in enumerate(zip(ref['events'],receipt['events'])) if a[0]!=b[0] or abs(a[1]-b[1])>1e-6]
        if not mismatches and len(ref['events'])!=len(receipt['events']):
            mismatches=[min(len(ref['events']),len(receipt['events']))]
        receipt['archive_comparison']={'same_physical_hash':ref['world']==receipt['world'],
          'first_different_event_zero_based':next(iter(mismatches),None),'expected_events':len(ref['events']),
          'delta_G_mbps':receipt['result']['G_mbps']-ref['result']['G_mbps'],
          'delta_R_pp':100*(receipt['result']['R']-ref['result']['R'])}
    dest.write_text(json.dumps(receipt,allow_nan=False))
    dest.replace(target)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split');p.add_argument('world',type=int)
    p.add_argument('arm');p.add_argument('scale',type=float)
    a=p.parse_args();r=run(a.split,a.world,a.arm,a.scale)
    print(r['protocol'],r['result']['G_mbps'],r.get('archive_comparison'),flush=True)
