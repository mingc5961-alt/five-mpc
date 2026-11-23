"""Actual and equal-coverage width. The latter is retrospective and never enters a replay."""
import json
import pickle
import numpy as np
from protocol import OUT,MODELS,KINDS,write
from fit_heads import predict
from features import K
from refine.tubes import scores
from routeclock.validity import finite_quantile

if __name__=='__main__':
    summary=[]
    for model in MODELS:
        with np.load(OUT/'features'/model/'validation.npz') as d:
            x,z,err=d['common'],d['special'],d['error']
            for kind in KINDS:
                head=pickle.loads((OUT/'heads'/model/(kind+'.pkl')).read_bytes())
                raw=predict(head,x,z if kind=='special' else None)
                s=scores(err,raw)
                for j,h in [(4,5),(6,10)]:
                    # The 1080th sorted value yields empirical 90% out of 1200 at this deadline.
                    q=np.sort(s[:,j])[int(np.ceil(.9*len(s)))-1]
                    w=np.rad2deg(raw[:,K[j+1]]*q).mean(0)
                    actual=np.rad2deg(raw[:,K[j+1]]*head['factors'][j]).mean(0)
                    summary.append(dict(model=model,kind=kind,h=h,cover=float(np.mean(s[:,j]<=head['factors'][j])),
                                equalcover=float(np.mean(s[:,j]<=q)),equalwidth=w.tolist(),width=actual.tolist()))
    write(OUT/'validation_widths.json',summary)
