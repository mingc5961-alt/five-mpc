"""Train the VECTOR velocity GRU with the causal 360-grid input used for replay."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from refine import models

ROOT = Path(__file__).resolve().parent
OBS = ROOT.parent / "refine_v1/observations"
UPSTREAM = next((ROOT / "work/vector").glob("drone_path_predictor_ros-*"))
HORIZONS = [0, 1, 3, 5, 9, 15, 19]  # half-second future indices
SEEDS = (7, 11, 19)

def features(times, positions, origin):
    end = times.searchsorted(origin, side="right")
    start = times.searchsorted(origin - 180.)
    times, positions = times[start:end], positions[start:end]
    grid = origin - np.arange(359, -1, -1) * .5
    idx = times.searchsorted(grid, side="right") - 1
    supported = idx >= 0
    velocities = np.zeros_like(positions)
    velocities[1:] = np.diff(positions, axis=0) / np.diff(times)[:, None]
    idx = idx.clip(min=0)
    observed = supported & (times[idx] > grid - .5)
    age = np.where(supported, np.minimum(180., grid - times[idx]) / 180., 1.)
    return np.column_stack((velocities[idx] * supported[:, None], supported, observed, age))

def read_split(split, center_only):
    rows = []
    files = sorted((OBS/split).glob("*.npz"))
    assert len(files) == (1200 if split == "train" else 400)
    for file in files:
        if center_only and int(file.stem) % 5 == 0:
            continue
        with np.load(file) as d:
            times, positions = d["times"], d["positions"]
            for t, truth in zip(d["origins"], d["truth"], strict=True):
                idx = times.searchsorted(t)
                assert times[idx] == t
                rows.append((features(times, positions, t), positions[idx], truth[25::25]))
    return tuple(torch.as_tensor(np.stack(x), dtype=torch.float32, device="cuda") for x in zip(*rows))

class Center(nn.Module):
    def __init__(self):
        super().__init__()
        src = UPSTREAM/"drone_path_predictor_ros/trajectory_predictor.py"
        author = models.external_module("vector_author", src)
        self.gru = author.VelocityPredictor2(3, 256, 3, 5)
        config = UPSTREAM/"config/5000_gz_dataset"
        self.gru.load_state_dict(torch.load(next(config.glob("*.pth")), weights_only=True, map_location="cpu"))
        self.mask_projection = nn.Linear(3, 3, bias=False)
        nn.init.zeros_(self.mask_projection.weight)
        with np.load(config/"vel_stats.npz") as d:
            self.register_buffer("mean", torch.tensor(d["input_mean"], dtype=torch.float32).reshape(3))
            matrix = torch.tensor(d["L_matrix"], dtype=torch.float32)
            self.register_buffer("whiten", matrix.T.contiguous())
            self.register_buffer("invert", torch.linalg.inv(matrix).T.contiguous())

    def forward(self, x):
        encoded = (x[..., :3] - self.mean) @ self.whiten + self.mask_projection(x[..., 3:])
        # Carry 180 seconds of state forward; truncate gradients through the prefix.
        with torch.no_grad():
            _, prefix_state = self.gru.gru1(encoded[:, :-20])
        _, hidden = self.gru.gru1(encoded[:, -20:], prefix_state)
        rep, _ = self.gru.gru2(x.new_zeros(len(x), 20, 256), hidden)
        return ((self.gru.fc(rep) @ self.invert + self.mean) * .5).cumsum(dim=1)

def angles(x):
    return torch.stack((torch.atan2(x[...,1],x[...,0]),torch.atan2(x[...,2],torch.linalg.vector_norm(x[...,:2],dim=-1))),dim=-1)

def score(net, data, ids):
    x, origin, truth = (v[ids] for v in data)
    pred = net(x) + origin[:,None]
    delta = angles(pred[:,HORIZONS]) - angles(truth[:,HORIZONS])
    return torch.linalg.vector_norm(torch.atan2(delta.sin(),delta.cos()),dim=-1).mean()

def write(path, content):
    temp = path.with_suffix(".tmp")
    temp.write_text(content)
    temp.replace(path)

def train(seed, trainset, val):
    folder = ROOT/"training/VECTOR"/str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"complete.json").exists():
        return
    torch.manual_seed(seed)
    net = Center().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=.0003)
    start = time.monotonic()
    best, stale = float("inf"), 0
    max_epochs, batch_size = 12, 32
    for epoch in range(1, max_epochs+1):
        net.train()
        batches = torch.randperm(len(trainset[0]), device="cuda").split(batch_size)
        for step, ids in enumerate(batches,1):
            opt.zero_grad(set_to_none=True)
            loss = score(net, trainset, ids)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if step == 1 or step % 5 == 0:
                write(folder/"progress.txt", f"第 {epoch}/{max_epochs} 轮，第 {step}/{len(batches)} 批，已用 {time.monotonic()-start:.0f} 秒")
        net.eval()
        with torch.no_grad():
            total = sum(float(score(net,val,ids))*len(ids) for ids in torch.arange(len(val[0]),device="cuda").split(batch_size))
        metric = total/len(val[0])
        if not np.isfinite(metric):
            raise ValueError("non-finite validation metric")
        with (folder/"curve.jsonl").open("a") as f:
            print(json.dumps(dict(epoch=epoch, validation_rad=metric, elapsed_s=time.monotonic()-start)), file=f)
        if metric < best:
            temp = folder/"best.partial"
            torch.save(net.state_dict(),temp)
            temp.replace(folder/"best.pt")
            best, stale = metric, 0
        else:
            stale += 1
        if stale >= 5 or time.monotonic()-start >= 3600:
            break
    with (folder/"best.pt").open("rb") as f:
        digest = hashlib.file_digest(f,"sha256").hexdigest()
    write(folder/"complete.json",json.dumps(dict(seed=seed, validation_rad=best, epochs=epoch, sha256=digest, artifact="center",peak_gpu_bytes=torch.cuda.max_memory_allocated())))
    write(folder/"progress.txt",f"中心训练完成，共 {epoch} 轮，最佳验证误差 {best:.5f} rad")

if __name__ == "__main__":
    trainset = read_split("train", True)
    validation = read_split("validation", False)
    for seed in SEEDS:
        train(seed, trainset, validation)
