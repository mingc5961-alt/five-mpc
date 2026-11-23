import numpy as np,pickle
from unittest.mock import patch
from core import ROOT,log
from learning import load,infer
from features import common,rc_features
from refine.data import ARCHIVE
from refine.samples import paths,records
from refine.tubes import errors
import fit_heads
def packed_rows(f,archive):
    with np.load(f) as data:rows=[{k:data[k][j] for k in data if k!='truth'} for j in range(len(archive))]
    for row,r in zip(rows,archive,strict=True):row['pathbase']=r['package'].centers['P'].astype('float32')
    return rows
def train():
    model=load('cpu')
    for split in ('train','validation','calibration'):
        rows=[]
        for f in paths(split):
            if split=='train' and int(f.stem)%5:continue
            a=pickle.loads((ARCHIVE/split/(f.stem+'.pkl')).read_bytes())
            packed=packed_rows(ROOT/'packed'/split/f.name,a);cs=infer(model,packed)
            for c,(h,t),r in zip(cs,records(f),a,strict=True):
                rows.append((int(f.stem),common(h,c),rc_features(r['package'],r['features']['P']),errors(c,t)))
        dest=ROOT/'features/joint';dest.mkdir(exist_ok=True,parents=True)
        np.savez(dest/(split+'.npz'),**dict(zip(('ids','common','special','error'),map(np.stack,zip(*rows)),strict=True)))
    with patch.object(fit_heads,'OUT',ROOT),patch.object(fit_heads,'KINDS',('special',)),patch.object(fit_heads,'log',log):fit_heads.fit('joint')
if __name__=='__main__':train()
