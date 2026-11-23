"""Upgrade only this campaign's active workers immediately after a durable per-cell receipt."""
import os,sys,time,signal
from pathlib import Path
out=Path(__file__).resolve().parent
eligible={}
for p in Path('/proc').glob('[0-9]*'):
    try:
        if (p/'cwd').resolve()!=out:continue
        a=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        if a[1:2]==['-u']:a=a[2:]
        else:continue
        if a[:1]!=['replay_route.py'] and a[:2]!=['other_models.py','replay']:continue
        path=(p/'fd/1').resolve()
        if not path.exists():continue
        eligible[int(p.name)]=(path,path.stat().st_size)
    except (OSError,IndexError):pass
while not (out/'native_check_CR-transfer.json').exists():time.sleep(4)
deadline=time.monotonic()+1200
while eligible and time.monotonic()<deadline:
    for pid,(path,pos) in list(eligible.items()):
        if not Path(f'/proc/{pid}').exists():eligible.pop(pid);continue
        with path.open('rb') as f:f.seek(pos);s=f.read()
        if '闭环'.encode() not in s:continue
        os.kill(pid,signal.SIGTERM)
        print('rotate at receipt',pid,path.name,flush=True)
        eligible.pop(pid)
    time.sleep(1)
