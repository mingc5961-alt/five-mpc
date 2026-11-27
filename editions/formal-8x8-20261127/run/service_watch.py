"""Keep queue and local-only progress server available after a coordinator exit."""
import subprocess,sys,time,fcntl
from base import *
def alive(p):
    try:
        pid=int(p.read_text());st=Path(f'/proc/{pid}/stat').read_text().split()
        return st[2]!='Z' and Path(f'/proc/{pid}/cwd').resolve()==BASE
    except (OSError,ValueError):return False
def spawn(name,argv,logname):
    with (BASE/'logs'/logname).open('ab',buffering=0) as f:
        p=subprocess.Popen(argv,cwd=BASE,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
    (BASE/(name+'.pid')).write_text(str(p.pid))
def main():
    lock=(BASE/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (BASE/'STOP').exists():
        if not alive(BASE/'http.pid'):spawn('http',[sys.executable,'-m','http.server','18765','--bind','127.0.0.1','--directory',str(BASE/'live')],'http.log')
        if not alive(BASE/'supervisor.pid'):
            s=json.loads((BASE/'live/status.json').read_text())['counts']
            if any(s.get(k) for k in ('running','ready','waiting')):
                spawn('supervisor',[str(BASE/'start.sh')],'supervisor.log')
        time.sleep(30)
if __name__=='__main__':main()
