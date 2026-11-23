"""Record genuine checkpoint outputs on one causal development history."""
import json
import hashlib
import numpy as np
import torch
from .models import ASSETS,BASE,ROOT,Chronos,Moirai,VectorNet,DEVICE,external_module
from .samples import records,paths

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    history,_=next(records(paths('train')[0]))
    result={}
    for name,cls in [('chronos',Chronos),('moirai',Moirai)]:
        pos,scale=cls().batch([history])[0]
        assert np.isfinite(pos).all() and np.isfinite(scale).all() and (scale>0).all()
        result[name]=dict(weight_sha256=sha(ASSETS/name/'model.safetensors'),shape=list(pos.shape),
                          raw_scale_rad_shape=list(scale.shape),first_future_position_m=pos[1].tolist(),first_scale_rad=scale[1].tolist())
    directory=next((ROOT/'upstream').glob('drone_path_predictor_ros-*'))
    author=external_module('vector_smoke',directory/'drone_path_predictor_ros/trajectory_predictor.py')
    folder=directory/'config/5000_gz_dataset'
    position=folder/'5000_pos_white_norm_256_5_0p5.pth'
    velocity=folder/'5000_vel_white_norm_256_5_0p5.pth'
    predictor=author.Predictor(str(position),str(velocity),str(folder/'pos_stats.npz'),str(folder/'vel_stats.npz'),
                               pos_hidden_dim=256,pos_num_layers=5,vel_hidden_dim=256,vel_num_layers=5,use_whitening=True)
    times=np.array([o.t for o in history]); xyz=np.array([o.position for o in history])
    idx=times.searchsorted(times[-1]-np.arange(19,-1,-1)*.1,side='right').clip(1,len(times))-1
    local=xyz[idx]-xyz[-1]
    p=predictor.predict_positions(local)
    v=predictor.predict_positions_from_velocity(local,.1)
    assert p.shape==(10,3) and v.shape==(11,3) and np.isfinite(p).all() and np.isfinite(v).all()
    result['vector_author']=dict(position_sha256=sha(position),velocity_sha256=sha(velocity),dt_s=.1,position_shape=list(p.shape),integrated_shape_including_origin=list(v.shape),
                                 velocity_path_last_offset_m=v[-1].tolist())
    (BASE/'intake/model_smoke.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
