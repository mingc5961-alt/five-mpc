"""Lightweight Linux cgroup, experiment process and NVIDIA device sampling."""
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent
RUNTIME=BASE/'runtime'
CG=Path('/sys/fs/cgroup')
GIB=2**30
TICKS=os.sysconf('SC_CLK_TCK')

def nvidia(query):
    try:
        value=subprocess.check_output(['nvidia-smi',query,'--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=3)
        return list(csv.reader(value.splitlines(),skipinitialspace=True))
    except (OSError,subprocess.SubprocessError):
        return []

def processes():
    entries={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv=(p/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            if len(argv)<2 or 'python' not in Path(argv[0]).name: continue
            if '-m' in argv:
                module=argv[argv.index('-m')+1]
                if not module.startswith(('refine.','runtime.')): continue
            elif argv[1]=='-c' and len(argv)>3 and 'from runtime.remaining_models import' in argv[2]:
                module='校准 '+argv[3]
            elif Path(argv[1]).name=='precompute_refine_worlds.py':
                module='确认世界预生成'
            elif Path(argv[1]).name=='prefetch_validation.py':
                module='验证世界预生成'
            else: continue
            # Limit attribution to this workspace.
            if (p/'cwd').resolve()!=Path('/data/coding/routeclock_refine'): continue
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            entries[int(p.name)]=dict(task=' '.join(argv[2:]) if '-m' in argv else module,
                tick=int(stat[11])+int(stat[12]),birth=stat[19],
                rss=int(stat[21])*os.sysconf('SC_PAGE_SIZE')/GIB)
        except (OSError,IndexError,ValueError): continue
    return entries

def cputicks():
    d=dict(line.split() for line in (CG/'cpu.stat').read_text().splitlines())
    return int(d['usage_usec'])/1e6

def measure(prev=None):
    gpu=nvidia('--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw')
    apps=nvidia('--query-compute-apps=pid,used_gpu_memory')
    gpumem={int(pid):int(mb) for pid,mb in apps if pid.strip().isdigit() and mb.strip().isdigit()}
    entries=processes()
    now=time.time()
    cpu=cputicks()
    if prev is None:
        time.sleep(.6)
        return measure((now,cpu,entries))
    dt=max(.001,now-prev[0])
    for pid,item in entries.items():
        last=prev[2].get(pid,{})
        item['cores']=max(0,(item['tick']-last['tick'])/TICKS/dt) if last.get('birth')==item['birth'] else None
        item['gpu_mib']=gpumem.get(pid,0) if gpu else None
    quota,period=(CG/'cpu.max').read_text().split()
    mem= int((CG/'memory.current').read_text())/GIB
    cap=(CG/'memory.max').read_text().strip()
    disk=shutil.disk_usage(BASE)
    data=dict(time=now,dt_s=dt,cgroup_cpu_cores=(cpu-prev[1])/dt,
              cpu_limit=float(quota)/int(period) if quota!='max' else None,
              experiment_cores=sum(item['cores'] or 0 for item in entries.values()),
              cgroup_memory_gib=mem,memory_limit_gib=int(cap)/GIB if cap!='max' else None,
              disk_free_gib=disk.free/GIB,disk_total_gib=disk.total/GIB,
              gpu_name=gpu[0][0] if gpu else None,gpu_pct=int(gpu[0][1]) if gpu else None,
              device_vram_mib=int(gpu[0][2]) if gpu else None,device_capacity_mib=int(gpu[0][3]) if gpu else None,
              experiment_gpu_mib=sum(v for k,v in gpumem.items() if k in entries) if gpu else None,
              gpu_power_w=float(gpu[0][4]) if gpu else None)
    return data, entries, (now,cpu,entries)
