"""Common causal history and real-weight forecast adapters (station-relative metres)."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file
from routeclock.foundation import ChronosAdapter
from routeclock.geometry import angular_jacobian
from routeclock.validity import EPS
from .data import ASSETS, BASE, ROOT

HS = np.arange(21) * .5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def external_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def grid(history, length):
    times = np.array([o.t for o in history])
    xyz = np.array([o.position for o in history])
    t = times[-1] - np.arange(length-1,-1,-1)*.5
    indices = np.searchsorted(times,t,side='right')-1
    valid = indices >= 0
    return xyz[np.maximum(0,indices)],valid

def vector_input(history):
    t = np.array([o.t for o in history])
    xyz = np.array([o.position for o in history])
    v = np.zeros_like(xyz)
    if len(t)>1:
        v[1:] = np.diff(xyz,axis=0)/np.diff(t)[:,None]
    indices = t.searchsorted(t[-1]-np.arange(19,-1,-1)*.5,side='right')-1
    return v[np.maximum(0,indices)].astype('float32')

def forecast(q, obs):
    # Input is [20 steps, xyz, q10/q50/q90]. Preserve median for both tube variants.
    if q.shape != (20,3,3) or not np.isfinite(q).all():
        raise ValueError('invalid quantile forecast')
    middle = q[:,:,1]
    pos = np.vstack([obs.position,middle])
    widths = np.max(abs(q-middle[:,:,None]),axis=-1)
    origin = np.sqrt(np.maximum(np.diag(obs.covariance),0))
    jac = np.abs(np.array([angular_jacobian(p) for p in pos]))
    raw = np.einsum('hij,hj->hi',jac,np.vstack([origin,widths]))
    return pos,np.maximum(EPS,raw)

class Chronos:
    def __init__(self):
        self.adapter = ChronosAdapter(str(DEVICE),str(ASSETS/'chronos'))

    @torch.inference_mode()
    def batch(self,histories):
        inputs = [self.adapter.inputs(h,[],None,None,None,False) for h in histories]
        qs,_ = self.adapter.pipeline.predict_quantiles([v[0] for v in inputs],prediction_length=20,
                quantile_levels=[.1,.5,.9],cross_learning=False,context_length=720)
        return [forecast(q.cpu().numpy().transpose(1,0,2)+offset[None,:,None],h[-1])
                for q,(_,offset),h in zip(qs,inputs,histories,strict=True)]

class Moirai:
    def __init__(self):
        source = next((ROOT/'upstream').glob('uni2ts-*/src'))
        sys.path[:0] = [str(ROOT/'upstream/extra'),str(source)]
        cls = external_module('routeclock_moirai',source/'uni2ts/model/moirai2/module.py').Moirai2Module
        self.net = cls(**json.loads((ASSETS/'moirai/config.json').read_text())).to(DEVICE).eval()
        self.net.load_state_dict(load_file(ASSETS/'moirai/model.safetensors'))

    @torch.inference_mode()
    def batch(self,histories):
        # Independent axis calls, 128 causal half-second steps, left padding excluded.
        positions,valid = zip(*(grid(h,128) for h in histories))
        offsets = np.array([h[-1].position for h in histories])
        arr = np.array(positions)-offsets[:,None,:]
        x = torch.as_tensor(arr.transpose(0,2,1).reshape(-1,8,16),device=DEVICE,dtype=torch.float32)
        mask = torch.as_tensor(np.repeat(valid,3,axis=0).reshape(-1,8,16),device=DEVICE)
        x = x.masked_fill(~mask,0)
        sample = mask.any(-1).long()
        clock = torch.arange(8,device=DEVICE).expand_as(sample)
        pred = self.net(x,mask,sample,clock,torch.zeros_like(sample),torch.zeros_like(sample).bool(),training_mode=False)
        pred = pred[:,-1].reshape(len(histories),3,4,9,16).permute(0,2,4,1,3).reshape(-1,64,3,9)
        quantiles = pred[:,:20,:,[0,4,8]].cpu().numpy()+offsets[:,None,:,None]
        return [forecast(q,h[-1]) for q,h in zip(quantiles,histories,strict=True)]

class VectorNet(nn.Module):
    def __init__(self):
        super().__init__()
        directory = next((ROOT/'upstream').glob('drone_path_predictor_ros-*'))
        author = external_module('routeclock_vector',directory/'drone_path_predictor_ros/trajectory_predictor.py')
        self.gru = author.VelocityPredictor2(3,256,3,5)
        config = directory/'config/5000_gz_dataset'
        self.gru.load_state_dict(torch.load(config/'5000_vel_white_norm_256_5_0p5.pth',weights_only=True,map_location='cpu'))
        with np.load(config/'vel_stats.npz') as data:
            self.register_buffer('mean',torch.tensor(data['input_mean'],dtype=torch.float32).reshape(3))
            matrix = torch.tensor(data['L_matrix'],dtype=torch.float32)
            self.register_buffer('whiten',matrix.T.contiguous())
            self.register_buffer('invert',torch.linalg.inv(matrix).T.contiguous())
        self.head = nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,2),nn.Softplus())

    def forward(self,v):
        _,hidden = self.gru.gru1((v-self.mean)@self.whiten)
        rep,_ = self.gru.gru2(v.new_zeros((len(v),20,256)),hidden)
        increments = (self.gru.fc(rep)@self.invert+self.mean)*.5
        return increments.cumsum(dim=1), self.head(rep)*.01

class Vector:
    def __init__(self,checkpoint=None):
        self.net = VectorNet().to(DEVICE).eval()
        self.net.load_state_dict(torch.load(checkpoint or BASE/'weights/vector7.pt',weights_only=True,map_location=DEVICE))

    @torch.inference_mode()
    def batch(self,histories):
        x = torch.as_tensor(np.stack([vector_input(h) for h in histories]),device=DEVICE)
        moves,scales = self.net(x)
        result = []
        for h,v,s in zip(histories,moves.cpu().numpy(),scales.cpu().numpy(),strict=True):
            o = h[-1]
            origin_scale = abs(angular_jacobian(o.position))@np.sqrt(np.maximum(0,np.diag(o.covariance)))
            result.append((np.vstack([o.position,o.position+v]),np.maximum(EPS,np.vstack([origin_scale,s]))))
        return result
