"""Measure the previously selected heads on the same validation anchors."""
import pickle
import numpy as np
from protocol import OUT,ROOT,MODELS,write,log
from features import K,EXPAND,rc_params,apply_gate,Advanced
from refine.samples import paths,records
from refine.data import ARCHIVE
from refine.tubes import scores,errors
from advanced_tubes import get_bundle

def baseline(model):
    with np.load(OUT/'features'/model/'validation.npz') as f:
        err=f['error'];ids=f['ids']
    if model=='RC':
        params=rc_params()
        factors=params['factors']['default'];raw=[]
        for file in paths('validation'):
            rows=pickle.loads((ARCHIVE/'validation'/(file.stem+'.pkl')).read_bytes())
            for r in rows:
                raw.append(params['head'].raw(r['package'],r['features']['P'])[EXPAND])
    elif model=='Chronos-2':
        engine=Advanced(model)
        bundle=get_bundle(engine.net,7)
        factors=bundle['factors']['native'];raw=[];pending=[]
        def flush():
            for (h,truth),(c,width) in zip(pending,engine.net.batch([p[0] for p in pending]),strict=True):
                np.testing.assert_allclose(errors(c,truth),err[len(raw)],rtol=0,atol=1e-8)
                raw.append(width)
            pending.clear()
        for file in paths('validation'):
            for r in records(file):
                pending.append(r)
                if len(pending)==16:flush()
        if pending:flush()
    else:
        with (ROOT/'tubes'/model/'7/bundle.pkl').open('rb') as f: bundle=pickle.load(f)
        factors=bundle['factors']['fixed']
        raw=np.broadcast_to(bundle['fixed'],(len(err),21,2))
    raw=np.asarray(raw)
    assert len(raw)==len(err)==1200 and len(set(ids))==400
    coverage=(scores(err,raw)<=factors).mean(0)
    width=np.rad2deg(raw[:,K[1:]]*factors[None,:,None]).mean(0)
    write(OUT/'original'/f'{model}.json',dict(validation=dict(n_routes=400,n_anchors=1200,
          coverage=coverage.tolist(),angular_halfwidth_deg=width.tolist()),center_5s_10s_deg=
          np.rad2deg(np.linalg.norm(err[:,[250,500]],axis=-1)).mean(0).tolist()))
if __name__=='__main__':
    for model in MODELS:baseline(model)
    log('四个归档参照管在相同 400 条验证轨迹上已补算双角前缀覆盖、实际角半宽与中心双角误差。')
