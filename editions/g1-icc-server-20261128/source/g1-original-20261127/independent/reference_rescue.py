"""Monitor incoming reference-world assets, validating the archive digest before unpickling."""
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from history import ROOT,PACKAGE,RUNTIME,ASSETS,REPO,REVISION,put
from physical_inputs import fingerprint,sha_file,ENGINE
from update_state import update
from audit_receipt import audit
DIR=ROOT/'recovery_01/reference_input'

def locate():
    expected=json.loads(subprocess.check_output(['git','-C',str(REPO),'show',f'{REVISION}:editions/g1-paid-cold-start-20261127/evidence/world_hashes.json']))['confirm0.world']
    dest=DIR/'confirm0.world'
    if dest.is_file() and sha_file(dest)==expected:return dest
    for base in (Path('/data/chen/refine-logs'),Path('/data/chen')):
        options=[base/'confirm0.world',base/'factorial_run/confirm0.world']
        for p in options:
            if p.is_file() and sha_file(p)==expected:
                shutil.copyfile(p,dest.with_suffix('.tmp'))
                if sha_file(dest.with_suffix('.tmp'))==expected:dest.with_suffix('.tmp').replace(dest);return dest
        for p in base.glob('*.zip'):
            if not any(s in p.name.lower() for s in ('world','physics','g1')):continue
            try:
                with zipfile.ZipFile(p) as z:
                    for entry in z.infolist():
                        if entry.filename.replace('\\','/').split('/')[-1]!='confirm0.world':continue
                        tmp=dest.with_suffix('.tmp')
                        with z.open(entry) as source,tmp.open('wb') as file:shutil.copyfileobj(source,file)
                        if sha_file(tmp)==expected:tmp.replace(dest);return dest
                        tmp.unlink(missing_ok=True)
            except (OSError,zipfile.BadZipFile):continue
    return None

def run():
    import pickle
    DIR.mkdir(parents=True,exist_ok=True)
    p=locate()
    if p is None:return
    with p.open('rb') as stream:w=pickle.load(stream)
    sha,components=fingerprint(w)
    if sha!='2d6ee19463199066323fb21187748993544e8ad5eff1f6d3eb294225dcdd4752':raise ValueError('serialized reference does not contain archived physics')
    put(p.with_suffix('.json'),dict(world=0,physical_sha256=sha,components=components,generator_sha256=sha_file(ENGINE/'routeclock/world.py'),
        source_sha256=sha_file(PACKAGE/'reference/editions/g1-paid-cold-start-20261127/evidence/work/factorial_data.py')))
    def replay(arm):
        out=DIR/f'anchor_{arm}_{time.time_ns()}'
        with (DIR/f'{arm}.log').open('ab') as log:
            subprocess.run([sys.executable,'-Bu',str(ROOT/'recovery_01/compat/with_physics.py'),str(p),str(PACKAGE),'replay_anchor','--arm',arm,'--repo',str(PACKAGE/'reference'),
                 '--cpu-root',str(ASSETS),'--runtime',str(RUNTIME),'--out',str(out)],check=True,stdout=log,stderr=log)
        verification=json.loads((out/'verification.json').read_bytes())
        if verification['status']!='PASS':raise ValueError('package strict verifier')
        actual=next(out.glob('run_confirm_*.json'))
        ref=REPO/'editions/g1-paid-cold-start-20261127/evidence/work/factorial_run'/actual.name
        a=audit(ref,actual);put(out/'full_ledger_audit.json',a)
        if (not a['physical_match'] or a['actual_events']!=a['expected_events'] or
                a['matching_event_pairs_1us']!=a['expected_events']):
            raise ValueError('complete archived event and world audit')
        name='g1' if arm=='G1' else 'chronos'
        put(ROOT/f'accounting/{name}_reference_recovery.json',dict(
            kind='historical original physics replay',arm=arm,world=0,scale=.45 if arm=='G1' else .75,
            status='PASS',complete_episodes=1,elapsed_s=verification['wall_s'],expected_events=a['expected_events'],
            matched_events=a['matching_event_pairs_1us'],ledger_seconds_delta=a['fields']['ledger_s']['delta'],
            physical_hash_match=True,expected_world=a['expected_world'],actual_world=a['actual_world'],
            verification=str(out/'verification.json'),comparison=str(out/'full_ledger_audit.json'),actual_path=str(actual)))
        return arm,str(out)
    with ThreadPoolExecutor(2) as pool:results=dict(pool.map(replay,['G1','C']))
    put(DIR/'summary.json',dict(status='PASS',world=sha,results=results))
    update(g1_hist='通过',g1_hist_completed=True,g1_hist_note='载入原始世界后，包内完整验收 PASS',chrono='原始世界回放通过',chrono_note='原始世界已核对')
if __name__=='__main__':run()
