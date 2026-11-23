from pathlib import Path
from datetime import datetime
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'RC进度与分支'
def log(s):
 line=f"{datetime.now():%F %T} RC候选里程 {s}"
 with (OUT.parent/'实时实验进度.txt').open('a') as f:print(line,file=f)
 print(line,flush=True)
