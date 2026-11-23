"""Keep scheduled cells running and publish the report after all checks pass."""
import os,sys,time,json,subprocess,datetime,fcntl
from pathlib import Path
OUT=Path(__file__).resolve().parent
os.chdir(OUT)
def counts():
    counts={g:0 for g in ['gpu','cpu','VECTOR','RAFT']}
    for f in (OUT/'cells').glob('*.json'):
        counts['cpu' if f.stem.endswith('RC-special') else 'gpu']+=1
    for f in (OUT/'other_cells').glob('*.json'):
        counts['VECTOR' if '_VECTOR-' in f.stem else 'RAFT']+=1
    return counts
def active():
    out={g:0 for g in ('gpu','cpu','VECTOR','RAFT')}
    for p in Path('/proc').glob('[0-9]*'):
        try:argv=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except OSError:continue
        for i,v in enumerate(argv):
            if Path(v).name=='replay_route.py' and argv[i+1:] in (['gpu'],['cpu']):out[argv[i+1]]+=1
            if Path(v).name=='other_models.py' and argv[i+1:] in (['replay','VECTOR'],['replay','RAFT']):out[argv[-1]]+=1
    return out
def launch(g):
    cmd=['replay_route.py',g] if g in ('gpu','cpu') else ['other_models.py','replay',g]
    with (OUT/'logs'/('supervised-'+g+'.log')).open('a') as out:
        return subprocess.Popen([sys.executable,'-u',*cmd],stdout=out,stderr=subprocess.STDOUT)
lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
maxima={'gpu':210,'cpu':30,'VECTOR':90,'RAFT':90}; error_count={g:0 for g in maxima}; children={}
prev=-1
previous_counts={g:0 for g in maxima}; cooldown={g:0. for g in maxima}
while True:
    status=counts(); now=active()
    if status==maxima:break
    for g in maxima:
        if status[g]>previous_counts[g]:
            error_count[g]=0; cooldown[g]=0.
    previous_counts=status.copy()
    if sum(status.values())!=prev:
        print(datetime.datetime.now().isoformat(timespec='seconds'),status,now,flush=True);prev=sum(status.values())
    for proc,g in list(children.items()):
        if proc.poll() is not None:
            if proc.returncode and proc.returncode!=-15:
                error_count[g]+=1
                cooldown[g]=time.monotonic()+min(240,10*2**min(error_count[g],4))
                print(datetime.datetime.now().isoformat(timespec='seconds'), g,'worker exit',proc.returncode,'retry after',int(cooldown[g]-time.monotonic()),flush=True)
            del children[proc]
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())
    total=sum(now.values())
    for group in sorted(maxima,key=lambda g:(maxima[g]-status[g])/(1+now[g]),reverse=True):
        if status[group]>=maxima[group] or cooldown[group]>time.monotonic():continue
        cap=(6 if group=='gpu' else 2 if group=='cpu' else 4)
        if now[group]>=cap or total>=9 or (group!='cpu' and used>(10300 if group=='gpu' else 11000)):continue
        if group not in ('gpu','cpu') and not (OUT/'route_heads'/'heads'/group/'special.pkl').exists():continue
        child=launch(group);children[child]=group;total+=1
        if group!='cpu':used+=1500 if group=='gpu' else 700
    time.sleep(40)
for g in ('VECTOR','RAFT'):
    if not (OUT/(g+'_中心逐机.csv')).exists():
        subprocess.run([sys.executable,'other_models.py','centers',g],check=True)
subprocess.run([sys.executable,'summary.py'],check=True)
