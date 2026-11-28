"""Cross-check archived coordinates on the restored runtime, apart from formal runs."""
import concurrent.futures as futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'package/g1-runtime-recovery'
RUNTIME = ROOT/'recovery_01/runtime'
ASSETS = ROOT/'recovery_01/chronos_assets'
REPO = ROOT.parent/'g1-server-handoff/reference/five-mpc'
OUT = ROOT/'recovery_01/historical_diagnostics'
EDITION = Path('editions/g1-paid-cold-start-20261127/evidence/work/factorial_run')
REVISION = '96b6e630e02c5fc709aef32e5a655b33ddf2463c'
sys.path.insert(0,str(PACKAGE/'tools'))
sys.path.insert(0,str(ROOT))
from experiment_env import configure,check_models
from audit_receipt import audit

def put(path,data):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def prepare_world(k):
    import factorial_data
    factorial_data.ROOT=OUT/'worlds'
    w=factorial_data.fresh_world('confirm',k)
    assert w.mission==k and w.namespace=='signed_g1_factorial_v1/confirm' and w.n==16
    return k

def compare(job):
    import grurun, factorial_data, replay_anchor
    k,arm,m=job
    name=f'run_confirm_{k}_{arm}_{m:.2f}.json'
    receipt=OUT/'receipts'/name
    reference=OUT/'pinned'/name
    results=OUT/'audits'/name
    if results.exists() and receipt.exists():
        old=json.loads(results.read_bytes())
        if (old['actual_sha256']==hashlib.sha256(receipt.read_bytes()).hexdigest() and
            old['expected_sha256']==hashlib.sha256(reference.read_bytes()).hexdigest()):
            return old
    mark=OUT/'active'/name
    put(mark,dict(pid=os.getpid(),started=time.time(),arm=arm))
    started=time.monotonic()
    try:
        factorial_data.ROOT=OUT/'worlds'
        with patch.object(grurun,'world',factorial_data.fresh_world),patch.object(grurun,'path',lambda *args:receipt):
            grurun.run(('confirm',k,arm,m))
        expected=json.loads(reference.read_bytes())
        actual=json.loads(receipt.read_bytes())
        with patch.object(replay_anchor,'ANCHOR_WORLD',expected['world']):
            differences=replay_anchor.verify(actual,expected)
        record=audit(reference,receipt)
        record.update(world=k,arm=arm,multiplier=m,seconds=time.monotonic()-started,
                      differences=differences,reference_revision=REVISION)
        put(results,record)
        return record
    finally:
        mark.unlink(missing_ok=True)

def main():
    for sub in ('worlds','active','receipts','pinned','audits'):
        (OUT/sub).mkdir(parents=True,exist_ok=True)
    jobs=[(k,a,m) for a,scales in [('C',(.75,1.)),('G1',(.45,.75,1.))]
          for k in range(8) for m in scales if (k,a,m) not in [(0,'C',.75),(0,'G1',.45)]]
    for k,a,m in jobs:
        name=f'run_confirm_{k}_{a}_{m:.2f}.json'
        r=subprocess.check_output(['git','-C',str(REPO),'show',f'{REVISION}:{EDITION/name}'])
        dst=OUT/'pinned'/name
        if dst.exists() and dst.read_bytes()!=r:raise ValueError(name)
        if not dst.exists():dst.write_bytes(r)
    check_models(PACKAGE/'reference',ASSETS,with_chronos=True)
    configure(ASSETS,RUNTIME)
    import grurun, factorial_data, torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ctx=multiprocessing.get_context('fork')
    start=time.time()
    put(OUT/'batch.json',dict(pid=os.getpid(),start=start,total=len(jobs),workers=len(jobs),phase='worlds'))
    with futures.ProcessPoolExecutor(8,mp_context=ctx) as pool:
        assert set(pool.map(prepare_world,range(8)))==set(range(8))
    put(OUT/'batch.json',dict(pid=os.getpid(),start=start,total=len(jobs),workers=len(jobs),phase='history'))
    with futures.ProcessPoolExecutor(len(jobs),mp_context=ctx) as pool:
        work=[pool.submit(compare,job) for job in jobs]
        records=[f.result() for f in futures.as_completed(work)]
    reused=[]
    for folder,arm,scale in [('anchor_g1','G1',.45),('anchor_c','C',.75)]:
        d=ROOT/'recovery_01'/folder
        r=json.loads((d/'full_ledger_audit.json').read_bytes())
        v=json.loads((d/'verification.json').read_bytes())
        reused.append(dict(r,world=0,arm=arm,multiplier=scale,differences=v['differences']))
    all_rows=records+reused
    assert len({(r['world'],r['arm'],r['multiplier']) for r in all_rows})==40
    summary=dict(new_runs=len(records),reused_anchors=len(reused),worlds=8,wall_s=time.time()-start,
        all_coordinates=len(all_rows),within_event_tolerance=sum(r['matching_event_pairs_1us']==r['actual_events']==r['expected_events'] for r in all_rows),
        matched_pairs_1us=sum(r['matching_event_pairs_1us'] for r in all_rows),expected_pairs=sum(r['expected_events'] for r in all_rows),
        matching_physics=sum(r['physical_match'] for r in all_rows),
        all_checks_except_world_passed=sum(not set(r['differences'])-{'world','world physical-array SHA-256'} for r in all_rows),
        other_differences=[dict(world=r['world'],arm=r['arm'],m=r['multiplier'],issues=[x for x in r['differences'] if x not in ('world','world physical-array SHA-256')]) for r in all_rows if set(r['differences'])-{'world','world physical-array SHA-256'}])
    put(OUT/'summary.json',summary)
    put(OUT/'batch.json',dict(pid=os.getpid(),start=start,total=len(jobs),workers=len(jobs),phase='done'))
    print(summary,flush=True)

if __name__=='__main__': main()
