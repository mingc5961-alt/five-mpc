"""One common validation origin per route for same-center action flips."""
import csv
import hashlib
import pickle
import sys
import numpy as np
from routeclock.config import load_config
from routeclock.radio import Radio
from routeclock.scheduler import choose_actions
from .data import BASE
from .network import frontier
from .samples import paths,records
from .tubes import MODELS,features,tree_scale,errors,scores,QUERY_H

def run(name):
    cfg=load_config();radio=Radio(cfg);model=MODELS[name]()
    with (BASE/'weights'/(name+'_tubes.pkl')).open('rb') as f:bundle=pickle.load(f)
    files=paths('validation');assert len(files)==400
    # 16 anchors per pseudo-network with identical input histories in both arms.
    # These are fixed-history action diagnostics distinct from network performance.
    out=[]
    for i in range(0,len(files),16):
        batch=[next(records(f)) for f in files[i:i+16]]
        predictions=model.batch([h for h,_ in batch])
        fronts={t:[] for t in ('native','lightgbm','fixed')}
        metrics=[]
        for path,(h,truth),(c,native) in zip(files[i:i+16],batch,predictions,strict=True):
            digest=hashlib.sha256(c.tobytes()).hexdigest()
            tree=tree_scale(bundle['tree'],features(h,c)[None])[0]
            row=dict(route=int(path.stem),origin=h[-1].t,center_sha256=digest)
            err=errors(c,truth)
            for key,raw in [('native',native),('lightgbm',tree),('fixed',bundle['fixed'])]:
                k=bundle['factors'][key]
                fronts[key].append(frontier(radio,c,raw,k))
                joint=scores(err[None],raw[None])[0]<=k
                for j,delta in enumerate(QUERY_H[1:]):
                    row[f'{key}_covered_{delta:g}s']=int(joint[j])
                    row[f'{key}_width_deg_{delta:g}s']=np.rad2deg(2*k[j]*raw[round(delta*2)]).mean()
            metrics.append(row)
        choices={t:choose_actions(f,.0031,.05/3) for t,f in fronts.items()}
        for j,row in enumerate(metrics):
            for key,actions in choices.items():
                a=actions[j];row[key+'_action']=f'{a.pair[0]}x{a.pair[1]}/{a.interval:g}' if a else 'unavailable'
            row['lightgbm_flip']=int(row['native_action']!=row['lightgbm_action'])
            row['fixed_flip']=int(row['native_action']!=row['fixed_action'])
        out.extend(metrics)
    folder=BASE/'e22_results';folder.mkdir(exist_ok=True)
    target=folder/(name+'_fixed_history.csv');tmp=target.with_suffix('.part')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
    tmp.replace(target)
if __name__=='__main__':run(sys.argv[1])
