"""Use separate process pools for GPU predictors and CPU route models."""
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
PROGRESS=Path('/data/chen/refine-logs/实时实验进度.txt')
def append(message):
    with PROGRESS.open('a') as fp:print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),message,file=fp)
def job(i):
    with (OUT/'cells'/f'{i}.log').open('a') as f:
        subprocess.run([sys.executable,str(OUT/'compare.py'),'cell',str(i)],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    from compare import plan
    jobs=plan()
    with (OUT/'计划.csv').open('w') as fp:
        wr=csv.writer(fp);wr.writerow(['id','model','world_index','world_set','tube','prewarm','admission','MCS_dB'])
        for i,(name,j,spec) in enumerate(jobs):wr.writerow([i,name,j,*spec])
    append(f'跨版本归因扩展。已冻结 {len(jobs)} 个闭环格，每项四个合成世界。比较冻结/微调、旧/新输入、原生/专用分位树、预热稀疏度、世界批次、动作到期及重新准入、I/P/RC；全部紧档。每场同时记两种到期规则的吞吐。另有独立的 16 场 MCS 对照和同历史预测诊断。')
    failures=[]
    with ThreadPoolExecutor(3) as cpu,ThreadPoolExecutor(3) as gpu:
        fs={(cpu if name in ('RC','I','P') else gpu).submit(job,i):i for i,(name,j,spec) in enumerate(jobs)}
        for k,f in enumerate(as_completed(fs),1):
            try:f.result()
            except Exception:failures.append(fs[f])
            if k%8==0 or k==len(jobs):append(f'跨版本闭环调度　已结束 {k}/{len(jobs)}，其中成功 {k-len(failures)}，需复查 {len(failures)}。')
    print('failed jobs',failures,flush=True)
    sys.exit(bool(failures))
