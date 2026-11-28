"""Run N01 on independently verified immutable Linux physical inputs, preserving archive status."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import runpy
import sys
import time
from unittest.mock import patch

from history import ROOT,PACKAGE,RUNTIME,put
from physical_inputs import paths,sha_file,fingerprint,World
from audit_receipt import audit
from freeze_n01 import main as freeze
from experiment_env import configure

BASE=ROOT/'paired_linux_v2'
CPU=PACKAGE/'gru4505'
OLD_CANARY=ROOT/'paired_linux_v1/canary/N01_development_1400_0.45.json'
REF=PACKAGE/'verification'/OLD_CANARY.name
SCALES=(.45,.60,.75,1.)

def load_checked(split,k):
    file,info=paths(split,k)
    meta=json.loads(info.read_bytes())
    if sha_file(file)!=meta['file_sha256']:raise ValueError(f'file hash {file}')
    with file.open('rb') as stream:world=pickle.load(stream)
    digest,parts=fingerprint(world)
    if (digest,parts,world.mission,world.namespace,world.groups)!=(meta['physical_sha256'],meta['components'],k,meta['namespace'],meta['groups']):
        raise ValueError(f'physical input {file}')
    return world,meta

def coordinate(k,m):
    output=BASE/'receipts'
    dest=output/f'N01_development_{k}_{m:.2f}.json'
    mark=BASE/'active'/dest.name
    put(mark,dict(pid=os.getpid(),started=time.time(),world=k,scale=m))
    try:
        w,meta=load_checked('development',k)
        if not dest.exists():
            constructor=inspect.signature(World.__init__);calls=[]
            def supply(self,*args,**kwargs):
                c=constructor.bind(self,*args,**kwargs);c.apply_defaults();a=c.arguments
                if not (a['cfg']==w.cfg and a['namespace']==w.namespace and a['mission']==w.mission and a['groups']==w.groups
                    and a['n'] in (None,w.n) and a['reusable'] is None and a['shape'] is None and not calls):
                    raise ValueError('original constructor differs from immutable bank')
                self.__dict__.update(w.__dict__);calls.append(True)
            script=PACKAGE/'tools/run_n01.py'
            argv=[str(script),'--repo',str(PACKAGE/'reference'),'--runtime',str(RUNTIME),'--cpu-root',str(CPU),
                  '--out',str(output),'--world',str(k),'--multiplier',str(m)]
            with patch.object(sys,'argv',argv),patch.object(World,'__init__',supply):
                runpy.run_path(str(script),run_name='__main__')
            if len(calls)!=1:raise ValueError('world not consumed exactly once')
        r=json.loads(dest.read_bytes())
        if not (r['world']==meta['physical_sha256'] and r['arm']=='G1' and r['split']=='development' and
                r['k']==k and r['m']==m and r['provenance']['groups']==meta['groups'] and
                r['provenance']['runner_sha256']==sha_file(PACKAGE/'tools/run_n01.py')):
            raise ValueError(f'wrong N01 coordinate {dest}')
        return dict(world=k,scale=m,receipt=str(dest))
    finally:mark.unlink(missing_ok=True)

def main():
    for sub in ['receipts','active']:(BASE/sub).mkdir(parents=True,exist_ok=True)
    if not (BASE/'protocol.json').exists():
        put(BASE/'protocol.json',dict(name='g1_linux_paired_v2',registered=time.time(),reason='Independent within-server pair after v1 cross-platform calibration exceeded the original G tolerance. Windows reference receipt is immutable and remains a separate comparison.',
            acceptance='Existing native canary vs separate run using stored independently regenerated physical arrays: exact event list, physics, full result and provenance. All worlds shared, hashed, held fixed. Model and runner package SHA unchanged.',
            method='Original run_n01.py and grurun.run. Only World construction is supplied from the locked bank after checking constructor arguments. First 1400 x0.45, then 31, freeze locally only.',
            no_cross_platform_equivalence_claim=True))
    import replay_anchor
    x,y=json.loads(OLD_CANARY.read_bytes()),json.loads(REF.read_bytes())
    a=audit(REF,OLD_CANARY)
    with patch.object(replay_anchor,'ANCHOR_WORLD',y['world']):
        a['differences']=replay_anchor.verify(x,y)
    put(BASE/'cross_platform_canary.json',a)
    for key in ['groups','group_rng_seed','namespace','effective_config','runtime_sha256','model_sha256','source_commit','runner_sha256']:
        if x['provenance'][key]!=y['provenance'][key]:raise ValueError(f'cross-platform configuration mismatch {key}')
    for k in ['G_mbps','R','bits','ledger_s']:
        if k not in a['fields']:raise ValueError(k)
    configure(CPU,RUNTIME)
    import grurun,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.time()
    put(BASE/'batch.json',dict(start=start,total=32,workers=1,pid=os.getpid(),phase='canary',submitted=1))
    first=coordinate(1400,.45)
    actual=Path(first['receipt'])
    candidate=json.loads(actual.read_bytes())
    if x!=candidate:raise ValueError('local independent whole receipt is not exactly equal')
    put(BASE/'local_canary.json',dict(status='PASS',independent_native_receipt_sha256=sha_file(OLD_CANARY),
        cached_input_receipt_sha256=sha_file(actual),physical_sha256=x['world'],events=len(x['events']),
        exact_world=True,exact_all_receipt_fields=True,cross_platform_reference_status='FAIL',
        archive_differences=a['differences'],independent_world_meta=str(paths('development',1400)[1])))
    jobs=[(k,m) for k in range(1400,1408) for m in SCALES if (k,m)!=(1400,.45)]
    put(BASE/'batch.json',dict(start=start,total=32,workers=31,pid=os.getpid(),phase='grid',submitted=32))
    receipts=[first]
    with ProcessPoolExecutor(31,mp_context=multiprocessing.get_context('fork')) as pool:
        receipts.extend(f.result() for f in as_completed([pool.submit(coordinate,k,m) for k,m in jobs]))
    put(BASE/'index.json',sorted(receipts,key=lambda r:(r['world'],r['scale'])))
    frozen=freeze(BASE/'index.json',BASE/'freeze')
    manifest=dict(protocol='g1_linux_paired_v2',cross_platform_reference_status='FAIL',
                  index_sha256=sha_file(BASE/'index.json'),freeze_sha256=sha_file(BASE/'freeze/freeze.json'),
                  physical_sha256=frozen['world_sha256'],K=frozen['K'],PM_ms=frozen['PM_ms'],complete=32)
    put(BASE/'summary.json',manifest)
    put(BASE/'batch.json',dict(start=start,total=32,workers=31,pid=os.getpid(),phase='done',submitted=32))
    print(manifest,flush=True)

if __name__=='__main__':main()
