from pathlib import Path
import os,json,hashlib,time
BASE=Path(__file__).resolve().parent
ARMS=['T8','T180','G8','G180','T8P','Cnative','IMM','G8P','VECTOR','RAFT','Czero','Cexternal','Joint07']
SCALES=[1.,.9,.75,.55,.35]
DEV=list(range(3));CONF=list(range(100,110))
HIST=[30,60,120,180]
NAMES=dict(zip(ARMS,['①短','①长','②短','②长','③','④','⑤','⑥','⑦V','⑦R','⑧','⑨','⑩']))
def js(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tag(spec):
    return '_'.join(str(spec[k]) for k in sorted(spec))
def worldfile(w):return BASE/'worlds'/f'{w}.pkl'
