import pickle
import sys
import numpy as np
from protocol import OUT,log
from features import Advanced,common,rc_features,rc_params,apply_gate
from refine.samples import paths,records
from refine.data import ARCHIVE
from refine.tubes import errors

def collect(model):
    engine=None if model=='RC' else Advanced(model)
    params=rc_params() if model=='RC' else None
    dest=OUT/'features'/model
    dest.mkdir(parents=True,exist_ok=True)
    for split in ('train','calibration','validation'):
        path=dest/(split+'.npz')
        if path.exists(): continue
        files=[p for p in paths(split) if split!='train' or int(p.stem)%5==0]
        assert len(files)==(240 if split=='train' else 400)
        ids=[];xs=[];zs=[];ys=[]
        def append(i,h,truth,c,z):
            ids.append(i);xs.append(common(h,c));zs.append(z);ys.append(errors(c,truth))
        batch=[]
        def flush():
            if batch:
                for (i,h,t),(c,z) in zip(batch,engine.extract([r[1] for r in batch]),strict=True):
                    append(i,h,t,c,z)
                batch.clear()
        for f in files:
            if engine is None:
                archived=pickle.loads((ARCHIVE/split/(f.stem+'.pkl')).read_bytes())
                for (h,truth),r in zip(records(f),archived,strict=True):
                    p=r['package']; base=r['features']['P']
                    assert p.origin==h[-1].t
                    p.diagnostics['recent_spacing_s']=np.mean(np.diff([v.t for v in h[-9:]])) if len(h)>1 else 0.
                    apply_gate(p,params['gate'],base,'RC','P')
                    append(int(f.stem),h,truth,p.centers['RC'],rc_features(p,base))
            else:
                for h,t in records(f):
                    batch.append((int(f.stem),h,t))
                    if len(batch)==16: flush()
        flush()
        assert len(set(ids))==len(files) and len(ids)==len(files)*(1 if split=='calibration' else 3) and np.isfinite(xs).all() and np.isfinite(zs).all() and np.isfinite(ys).all()
        tmp=path.with_name('scratch.npz')
        np.savez(tmp,ids=ids,common=xs,special=zs,error=ys)
        tmp.replace(path)
        log(f'{model} {split} 提取完毕，{len(files)} 条轨迹、{len(ids)} 个锚点，每个期限公共 {xs[0].shape[1]} 个字段，专用 {zs[0].shape[1]} 个字段（VECTOR 压缩前）。')
if __name__=='__main__': collect(sys.argv[1])
