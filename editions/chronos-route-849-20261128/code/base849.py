import hashlib, json, os, sys, tempfile, subprocess
from pathlib import Path
for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[var]='1'
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parents[1]
OLD=Path('/data/coding/g1-original-20261127')
SRC=Path('/data/coding/.publish-g1-icc-20261128')
ED=SRC/'editions/g1-icc-server-20261128'
sys.path.insert(0,str(OLD/'independent'))
from policy_env import initialize
initialize()
import closed, paid_full as pf, numpy as np
from paired_grid import load_checked
from history import PACKAGE,RUNTIME,ASSETS
from physical_inputs import sha_file,fingerprint,load_config,GROUPS,World
from aoi_reference import QuotaGrid
from predictors import Model
P=json.loads((OUT/'protocol.json').read_bytes())
grid=QuotaGrid(service_us=180000000,budget_us=3000000,fix_us=3100,users=16,quota=849)
cal=[[grid.start_us(i),grid.start_us(i)+3100] for i in grid.fixed_indices()]
def sha(x):return hashlib.sha256(x).hexdigest()
def arrhash(a):return sha(np.ascontiguousarray(a).tobytes())
def compact(x):return json.dumps(x,separators=(',',':'),allow_nan=False).encode('ascii')
def atom(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    with os.fdopen(fd,'wb') as f:f.write(data)
    os.replace(tmp,path)
def put(path,data):atom(path,json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False).encode())
def get(path):return json.loads(Path(path).read_bytes())
def codehash():return sha(compact({p.name:sha_file(p) for p in sorted((OUT/'code').glob('*.py'))}))
def world(split,k):
    if split!='confirmation':return load_checked(split,k)
    from design849 import newworld
    return newworld(k)
def verified_cal():
    data=compact(cal)
    assert len(cal)==849 and data==(OUT/'package/calendar_849.json').read_bytes() and sha(data)==P['scene']['calendar_sha256']
    atom(OUT/'E2/calendar_849.json',data)
    return sha(data)
def progress(msg):atom(OUT/'实时进度.txt',('Chronos-849 正式实验实时进度\n'+__import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds')+'\n'+msg+'\n结果目录 '+str(OUT)+'\n').encode())
def preflight():
    from history import check_models
    from learned_matrix import selected_receipt,policy
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SRC,text=True).strip()==P['source_commit']
    ch=verified_cal()
    paths=check_models(PACKAGE/'reference',ASSETS,with_chronos=True)
    files={str(v):sha_file(Path(v)) for v in paths.values() if Path(v).is_file()}
    for p in [ASSETS/'chronos_snapshot/model.safetensors',*sorted((OUT/'package').iterdir()),Path('/data/chen/refine-logs/Chronos-849-server-plan.zip')]:
        if p.is_file():files[str(p)]=sha_file(p)
    for seed in [17,29,41]:
        checkpoint=Path(get(selected_receipt('fixed',seed))['model']);files[str(checkpoint)]=policy('fixed',seed)[1]
    w,wm=world('development',1400)
    net=Model(w,'Chronos')
    import grurun,grudata,torch,platform
    modules={str(Path(m.__file__).resolve()):sha_file(Path(m.__file__)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and str(m.__file__).startswith('/data/coding/')}
    put(OUT/'manifest.json',dict(source_repository=str(SRC),commit=P['source_commit'],files=files,paths=paths,modules=modules,grid=pf.GRID.tolist(),factors=np.asarray(net.factors).tolist(),world1400=wm,config=w.cfg,config_sha=sha(compact(w.cfg)),calendar_sha256=ch,device='cpu',python=sys.executable,torch=torch.__version__,platform=platform.platform(),logical_cpus=len(os.sched_getaffinity(0)),code_sha256=codehash()))
    progress('已校验版本、权重、开发世界 1400 和 849 笔预约表。正在重放旧开发锚点及准备单世界新路径。')
def anchor():
    import runpy
    from unittest.mock import patch
    import paired_grid
    output=OUT/'audits/anchor';output.mkdir(exist_ok=True)
    script=PACKAGE/'tools/run_n01.py'
    argv=[str(script),'--repo',str(PACKAGE/'reference'),'--runtime',str(RUNTIME),'--cpu-root',str(ASSETS),'--out',str(output),'--world','1400','--multiplier','0.45']
    # Independent native runner re-generates the world from its own seed.
    with patch.object(sys,'argv',argv):runpy.run_path(str(script),run_name='__main__')
    actual=get(output/'N01_development_1400_0.45.json');ref=get(paired_grid.OLD_CANARY)
    for key in ('result','world','events','k','m','arm'):assert actual[key]==ref[key],key
    put(OUT/'audits/legacy.json',dict(pass_exact=True,events=len(actual['events']),world=actual['world'],reference=str(paired_grid.OLD_CANARY),reference_sha=sha_file(paired_grid.OLD_CANARY),receipt_sha=sha_file(output/'N01_development_1400_0.45.json')))
if __name__=='__main__':{'preflight':preflight,'anchor':anchor}[sys.argv[1]]()
