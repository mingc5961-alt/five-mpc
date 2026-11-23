import json,subprocess,time,sys
from pathlib import Path
from statistics import mean
from collections import Counter
base=Path('/data/chen/routeclock_runs/refine_v1')
def stat():
 return {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}
start=stat();t=time.monotonic();gpu=[];power=[];used=[]
for _ in range(int(sys.argv[1])):
 r=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')
 gpu.append(int(r[0]));power.append(float(r[1]));used.append(int(r[2]));time.sleep(1)
end=stat();dt=time.monotonic()-t
print(json.dumps(dict(duration_s=dt,cpu_cores=(end['usage_usec']-start['usage_usec'])/1e6/dt,gpu_mean=mean(gpu),gpu_min=min(gpu),gpu_max=max(gpu),gpu_power_w=mean(power),vram_mib=max(used),throttled_pct=100*(end['nr_throttled']-start['nr_throttled'])/max(1,end['nr_periods']-start['nr_periods']),io_pressure=Path('/sys/fs/cgroup/io.pressure').read_text().splitlines()[0])))
