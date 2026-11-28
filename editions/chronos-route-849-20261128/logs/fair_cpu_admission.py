"""Operational CPU admission only. Frozen scientific code and receipts are read-only here."""
import json,os,signal,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def event(data):
    with (root/'logs/cpu_fairness.jsonl').open('a') as h:h.write(json.dumps(dict(timestamp=time.time(),**data))+'\n')
def current(tag):
    owners={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if b'code/overflow849.py' not in args or (b'--supplement' in args)!=(tag=='surplus'):continue
            children=[int(x) for x in (p/'task'/p.name/'children').read_text().split()]
            if children and (p/'stat').read_text().split()[2]!='T':owners[int(p.name)]=children
        except OSError:pass
    return owners
def grant(tag,pid,names):
    stat=Path(f'/proc/{pid}/stat');identity=stat.read_text().split()[21];limit=16 if tag=='surplus' else 10
    permit=root/f'logs/{tag}_enabled.txt';temp=root/f'logs/{tag}_fairness.tmp';start=time.time()
    os.kill(pid,signal.SIGSTOP);event(dict(action='grant_cpu',coordinator=tag,pid=pid,targets=names))
    try:
        while time.time()-start<140 and not (root/'report_receipt.json').exists():
            temp.write_text(str(limit));temp.replace(permit);time.sleep(5)
    finally:
        if stat.is_file() and stat.read_text().split()[21]==identity:os.kill(pid,signal.SIGCONT)
        event(dict(action='release_cpu',coordinator=tag,pid=pid,duration_s=time.time()-start))
def main():
    last={'overflow':0.,'surplus':0.}
    while not (root/'report_receipt.json').exists():
        time.sleep(15)
        special=root/'audits/final_common_lane_priority.json'
        if special.exists() and json.loads(special.read_text())['status']=='applied':continue
        for tag in sorted(last,key=last.get):
            if time.time()-last[tag]<250:continue
            for pid,children in current(tag).items():
                old=[]
                for f in (root/'active_closed/confirmation').glob('*.json'):
                    try:
                        r=json.loads(f.read_text());q=Path(f"/proc/{r['pid']}/wchan")
                        if r['pid'] in children and time.time()-f.stat().st_mtime>600 and q.read_text()=='hrtimer_nanosleep':old.append(f.name)
                    except (OSError,ValueError):pass
                if old:
                    grant(tag,pid,old);last[tag]=time.time();break
            else:continue
            break
if __name__=='__main__':main()
