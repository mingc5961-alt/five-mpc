from pathlib import Path
from datetime import datetime
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=Path('/data/chen/routeclock_runs/confirm_0db')
PRIOR=OUT.parent/'三先进模型LightGBM对比'
sys.path[:0]=[str(ROOT),str(ROOT/'work/code/routeclock_refine')]
MODELS=('RC','Chronos-2','VECTOR','RAFT')
KINDS=('common','special')
BUDGETS=('T','M','L')
NS='develop_0db_20261122/v1'
SEED=7

def log(message):
    with (OUT.parent/'实时实验进度.txt').open('a') as f:
        print(datetime.now().strftime('%F %T'), '专用尺度管实验',message,file=f)
    print(message,flush=True)

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False))
    tmp.replace(path)
