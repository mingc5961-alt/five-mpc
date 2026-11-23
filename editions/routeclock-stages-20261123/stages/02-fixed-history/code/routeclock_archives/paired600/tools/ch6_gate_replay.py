"""Counterfactual center at identical issued histories, actions and data times."""
import argparse
import json
import pickle
import sys
from collections import defaultdict,deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routeclock.geometry import angles,angle_error
from routeclock.config import dump_json


def replay(folder,bundle):
    with bundle.open('rb') as f:gate=pickle.load(f)['gates']['RC']
    clock=defaultdict(lambda:deque(maxlen=9))
    predictions={}
    for p in json.loads((folder/'predictions.json').read_text()):
        uid,origin=p['uav'],p['origin']
        history=clock[uid]
        history.append(origin)
        spacing=(history[-1]-history[0])/max(1,len(history)-1)
        package=SimpleNamespace(diagnostics={'recent_spacing_s':spacing})
        x=gate.inputs(np.array(p['features']),package)
        imm=np.array(p['experts']['I'])
        center=gate.predict(x,np.array(p['experts']['P']),imm)
        predictions[uid,origin]=(center,imm,np.array(p['h']))
    sums=np.zeros((3,4))
    for e in json.loads((folder/'events.json').read_text()):
        if 'prediction_age' not in e:continue
        age=e['prediction_age']
        center,imm,hs=predictions[e['uav'],e['prediction_origin']]
        truth=np.array(e['pointing'])+e['actual_angle_error']
        residual=[np.array(e['actual_angle_error'])]
        for xyz in (center,imm):
            point=[np.interp(age,hs,axis) for axis in xyz.T]
            residual.append(angle_error(angles(point),truth))
        sums[int(age>1)+int(age>2)]+=e['data_s']*np.r_[1.,np.mean(np.square(residual),axis=1)]
    out=[]
    for label,(seconds,*errors) in zip(['0-1','1-2','2-10'],sums):
        rmse=np.rad2deg(np.sqrt(np.array(errors)/seconds))
        out.append(dict(age_bin=label,data_s=seconds,G0=rmse[0],G1=rmse[1],I=rmse[2]))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('folder','bundle','out'):p.add_argument(name,type=Path)
    a=p.parse_args()
    dump_json(a.out,replay(a.folder,a.bundle))
