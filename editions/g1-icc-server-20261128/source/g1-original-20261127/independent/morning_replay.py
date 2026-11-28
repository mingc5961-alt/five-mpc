"""Rerun two preselected anchors from cold processes on the morning of the handoff."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import subprocess
import sys
from zoneinfo import ZoneInfo

from history import ASSETS,PACKAGE,ROOT,RUNTIME,put
from audit_receipt import audit
from physical_inputs import sha_file
from slots import compute_slot

OUT=ROOT/'accounting/morning_replay'
def check(arm):
    name='N01_development_1400_0.45.json' if arm=='G1' else 'run_confirm_0_C_0.75.json'
    first=(ROOT/'paired_linux_v2/receipts' if arm=='G1' else ROOT/'recovery_01/anchor_c')/name
    dest=OUT/arm/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tool='run_n01' if arm=='G1' else 'replay_anchor'
    args=['--world','1400','--multiplier','0.45'] if arm=='G1' else ['--arm','C']
    with compute_slot(),(dest.parent/(dest.name+'.log')).open('wb') as f:
        proc=subprocess.run([sys.executable,'-Bu',str(PACKAGE/f'tools/{tool}.py'),
            '--repo',str(PACKAGE/'reference'),'--runtime',str(RUNTIME),'--cpu-root',str(ASSETS),
            '--out',str(dest),*args],stdout=f,stderr=subprocess.STDOUT)
    receipt=dest/name
    if not receipt.is_file():raise RuntimeError(f'{arm} exited with {proc.returncode}, see {dest}.log')
    result=audit(first,receipt)
    if arm=='C':original=json.loads((dest/'verification.json').read_bytes())['status']
    else:
        from replay_anchor import verify
        from unittest.mock import patch
        ref=json.loads((PACKAGE/'verification'/name).read_bytes())
        with patch('replay_anchor.ANCHOR_WORLD',ref['world']):
            original='FAIL' if verify(json.loads(receipt.read_bytes()),ref) else 'PASS'
    same=sha_file(first)==sha_file(receipt)
    put(dest/'replication_audit.json',result)
    return dict(arm=arm,local_entire_receipt_equal=same,archived_verification=original,
        matched_local_events=result['matching_event_pairs_1us'],local_events=result['expected_events'],
        local_physics_equal=result['physical_match'],first_sha256=result['expected_sha256'],new_sha256=result['actual_sha256'],
        original=str(first),cold_replay=str(receipt))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(check,['G1','C']))
    put(OUT/'summary.json',dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        preselected_worlds=True,local_replication='PASS' if all(r['local_entire_receipt_equal'] for r in results) else 'FAIL',
        arms=results,additional_formal_endpoints=0))

if __name__=='__main__':
    main()
