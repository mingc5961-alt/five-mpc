"""Retain the frozen weights, split fingerprints and feature schema used by this run."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata as md
import numpy as np
from protocol import ROOT,OUT,PRIOR,MODELS,NS,SEED,write

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def grouped(files):
    h=hashlib.sha256()
    for f in files:
        h.update(f.name.encode());h.update(bytes.fromhex(digest(f)))
    return h.hexdigest()

def audit():
    obs=ROOT.parent/'refine_v1/observations'
    splits={}
    for name in ['train','calibration','validation']:
        paths=sorted((obs/name).glob('*.npz'))
        if name=='train':
            paths=[f for f in paths if int(f.stem)%5==0]
        ids=[int(f.stem) for f in paths]
        blocks=[]
        for m in MODELS:
            with np.load(OUT/'features'/m/(name+'.npz')) as data:
                blocks.append(data['ids'])
        assert all(np.array_equal(blocks[0],b) for b in blocks[1:])
        assert sorted(set(blocks[0]))==ids
        splits[name]={'route_ids':ids,'anchors':len(blocks[0]),'input_files_sha256':grouped(paths)}
    head_sha={m:{k:digest(OUT/'heads'/m/f'{k}.pkl') for k in ['common','special']} for m in MODELS}
    weights={m:digest(ROOT/'training'/m/'7/best.pt') for m in MODELS if m!='RC'}
    weights['RC_classical_bundle']=digest(PRIOR/'cache/classical.pkl')
    for m in MODELS[1:]:
        assert weights[m]==json.loads((ROOT/'training'/m/'7/complete.json').read_text())['sha256']
    source={p.name:digest(p) for p in OUT.glob('*.py')}
    write(OUT/'复现实验配置.json',dict(namespace=NS,worlds=list(range(10)),center_seed=SEED,
          budgets_airtime_s={'T':3,'M':4.5,'L':9},service_s=180,warm_s=180,users=16,mcs_margin_db=0,
          feedback_cost_s=.0031,score_interval_s=.02,geom_loss_db=3,maximum_deadline_s=10,
          center_sha256=weights,head_sha256=head_sha,split=splits,source_sha256=source,
          python=platform.python_version(),versions={k:md.version(k) for k in ['torch','numpy','lightgbm','scikit-learn']}))
if __name__=='__main__':audit()
