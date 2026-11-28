"""Measure CPU and proportional memory of this independent experiment process tree."""
import csv,datetime,os,time
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];HZ=os.sysconf('SC_CLK_TCK')
def sample():
    cpus={};pss=0;rss=0
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if Path(os.readlink(entry/'cwd'))!=OUT or (entry/'comm').read_text().strip() not in ('python','python3'):continue
            s=(entry/'stat').read_text().split();key=(entry.name,s[21]);cpus[key]=(int(s[13])+int(s[14]))/HZ
            mem={l.split(':')[0]:int(l.split()[1]) for l in (entry/'smaps_rollup').read_text().splitlines()[1:] if ':' in l}
            pss+=mem['Pss']*1024;rss+=mem['Rss']*1024
        except (OSError,ValueError,KeyError):pass
    stat=dict(v.split() for v in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    return time.monotonic(),cpus,pss,rss,int(stat['usage_usec'])*1e-6
def main():
    dest=OUT/'logs/cpu_memory.csv';prev=sample()
    with dest.open('a') as handle:
        z=csv.writer(handle)
        if handle.tell()==0:z.writerow(['utc','phase','processes_observed','study_cpu_cores','cgroup_cpu_cores','study_pss_bytes','sum_rss_bytes_with_shared_pages','cgroup_memory_bytes'])
        while not (OUT/'report_receipt.json').exists():
            time.sleep(15);now=sample();dt=now[0]-prev[0]
            cpu=sum(v-prev[1].get(k,v) for k,v in now[1].items())/dt
            phase='confirmation' if (OUT/'audits/final-smoke.json').exists() else 'final-smoke' if (OUT/'design/frozen.json').exists() else 'development'
            z.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),phase,len(now[1]),round(cpu,3),round((now[4]-prev[4])/dt,3),now[2],now[3],int(Path('/sys/fs/cgroup/memory.current').read_text())]);handle.flush();prev=now
if __name__=='__main__':main()
