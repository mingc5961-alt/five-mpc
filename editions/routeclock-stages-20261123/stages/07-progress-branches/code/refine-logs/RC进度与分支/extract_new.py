from concurrent.futures import ProcessPoolExecutor
import sys,pickle
from core import ROOT,log,pack
import numpy as np
from refine.samples import records,paths
from refine.data import ARCHIVE
from routeclock.predictors import PositionIMM,RouteMemory,apply_gate
from routeclock.config import load_config
GATE=pickle.loads((ROOT.parent/'三先进模型LightGBM对比/cache/classical.pkl').read_bytes())['RC']['gate']
CFG=load_config()
def extract(path):
    split=path.parent.name
    dest=ROOT/'packed'/split/path.name
    if dest.exists():return
    imm=PositionIMM();mem=RouteMemory(CFG['models']['route']);prev=0;rows=[]
    archive=pickle.loads((ARCHIVE/split/(path.stem+'.pkl')).read_bytes())
    for (h,y),r in zip(records(path),archive,strict=True):
        for o in h[prev:]:imm.observe(o);mem.update(o,imm.state()[0][3:])
        prev=len(h);p=r['package'];p.diagnostics['recent_spacing_s']=np.mean(np.diff([o.t for o in h[-9:]]))
        assert p.origin==h[-1].t
        md=p.diagnostics['memory']
        if 's' in md:np.testing.assert_allclose([mem.s,mem.active],[md['s'],md['active']],atol=1e-5)
        apply_gate(p,GATE,r['features']['P'],'RC','P')
        row=pack(h,mem,imm,p);row['truth']=y[::25].astype('float32');rows.append(row)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez(tmp,**{k:np.stack([r[k] for r in rows]) for k in rows[0]});tmp.replace(dest)
if __name__=='__main__':
    jobs=[]
    for split in ('train','validation','calibration'):
        (ROOT/'packed'/split).mkdir(exist_ok=True,parents=True);jobs+=paths(split)
    with ProcessPoolExecutor(int(sys.argv[1])) as pool:
        for j,_ in enumerate(pool.map(extract,jobs,chunksize=2),1):
            if j%100==0:log(f'因果输入重建 {j}/{len(jobs)} 条路线，逐锚核对当前片段及弧长。')
