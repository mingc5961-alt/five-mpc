"""Reconstructed 8-second masked GRU, with explicit Linux-refit checkpoints."""
from functools import lru_cache
import json
import numpy as np
import torch
from torch import nn
from server_paths import RUN, ROOT

GRID = np.arange(21)*.5
CHECKPOINT = RUN/'models'/'short_refit.pt'


class ResidualGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16,24)
        self.gru = nn.GRU(24,24,batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(24),nn.Linear(24,60))
        self.register_buffer('mean',torch.zeros(7))
        self.register_buffer('std',torch.ones(7))

    def forward(self, x, mask):
        z = ((x-self.mean)/self.std).clamp(-6,6)*mask[...,None]
        tokens = torch.cat([z,mask[...,None].to(x.dtype)],dim=-1).reshape(-1,4,16)
        _,h = self.gru(self.embed(tokens))
        return self.head(h[-1]).reshape(-1,20,3)


@lru_cache(maxsize=1)
def load_residual():
    checkpoint = torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    if checkpoint['protocol'] != 'linux_refit_v1':
        raise ValueError('unrecognized reconstructed checkpoint')
    net = ResidualGRU()
    net.load_state_dict(checkpoint['state'])
    return net.eval()


class Cold:
    def __init__(self, mode='G1'):
        if mode not in {'G1','G','Static','NoRoute','NoGRU','S'}:
            raise ValueError(f'unsupported cold predictor {mode}')
        self.mode = mode
        params = json.loads((ROOT/'reference/models/G1/fusion.json').read_bytes())
        self.prior = np.interp(GRID,[1,2,5,10],params['horizon'])
        self.net = load_residual() if mode!='NoGRU' else None

    def residual(self, history):
        if self.net is None:
            return np.zeros((20,3))
        from grudata import encode
        x,mask = encode(history[-32:])
        with torch.inference_mode():
            return self.net(torch.from_numpy(x)[None],torch.from_numpy(mask)[None])[0].numpy()
