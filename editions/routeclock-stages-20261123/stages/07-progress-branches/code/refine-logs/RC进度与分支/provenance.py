from core import ROOT,log
import hashlib,json,sys,platform
import numpy,torch,lightgbm
old=ROOT.parent/'专用尺度管实验'
prior=ROOT.parent/'微调Chronos路线对照'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((old/'复现实验配置.json').read_text())
files=[*(ROOT/(m+'.pt') for m in ['geometry','branch','progress','joint']),ROOT/'heads/joint/special.pkl',
       *(old/'heads'/m/'special.pkl' for m in ['RC','Chronos-2','RAFT','VECTOR']),prior/'fitted.pkl',prior/'VECTOR_adapter.pt',prior/'RAFT_adapter.pt',
       *(ROOT/f for f in ('learning.py','core.py','replay_new.py','heads_new.py','extract_new.py','evaluate.py','ceiling.py'))]
obj={'冻结资产与实现':{str(p.relative_to(ROOT.parent)):sha(p) for p in files},'原中心指纹':cfg['center_sha256'],
     '确认世界':list(range(40,50)),'世界命名空间':'confirm_0db_20261122/v1','GPU':torch.cuda.get_device_name(0),'Python':platform.python_version(),
     'torch':str(torch.__version__),'numpy':numpy.__version__,'lightgbm':lightgbm.__version__,'训练':json.loads((ROOT/'training.json').read_text()),
     '模型':{m:{k:v for k,v in torch.load(ROOT/(m+'.pt'),weights_only=True,map_location='cpu').items() if k!='state'} for m in ('geometry','progress','branch','joint')}}
(ROOT/'冻结配置.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
