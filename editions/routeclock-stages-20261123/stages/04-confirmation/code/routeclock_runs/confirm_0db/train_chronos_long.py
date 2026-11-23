"""Full-parameter Chronos-2 fine-tuning on grouped XYZ targets."""
from pathlib import Path
import time
import json

import numpy as np
import torch
from chronos import Chronos2Pipeline
from train_vector_long import HORIZONS, OBS, ROOT, SEEDS, angles, write

CHECKPOINT = Path("/data/chen/routeclock_inputs_20261120/chronos")

def context(times, positions, anchor):
    grid = anchor - np.arange(359,-1,-1) * .5
    idx = times.searchsorted(grid, side="right") - 1
    pos = positions[idx.clip(min=0)]
    mask = (idx >= 0) & (times[idx.clip(min=0)] > grid-.5)
    return np.where(mask[:,None], pos-pos[-1], 0), mask, pos[-1]

def split_data(split):
    rows = []
    files = sorted((OBS/split).glob("*.npz"))
    assert len(files) == (1200 if split == "train" else 400)
    for file in files:
        if split=="train" and int(file.stem)%5==0:
            continue
        with np.load(file) as d:
            times, positions = d["times"], d["positions"]
            for anchor, truth in zip(d["origins"],d["truth"],strict=True):
                x, mask, origin = context(times,positions,anchor)
                rows.append((x, mask, origin, truth[25::25]))
    return tuple(torch.as_tensor(np.stack(field), device="cuda", dtype=torch.float32) for field in zip(*rows))

def forward(model, data, ids, fitting):
    x, mask, origin, target = (field[ids] for field in data)
    groups = torch.arange(len(x),device=x.device).repeat_interleave(3)
    response = model(context=x.transpose(1,2).reshape(-1,360), context_mask=mask.repeat_interleave(3,dim=0),
                     group_ids=groups,num_output_patches=2,
                     future_target=(target-origin[:,None]).transpose(1,2).reshape(-1,20) if fitting else None)
    if fitting:
        return response.loss
    median = list(model.chronos_config.quantiles).index(.5)
    xyz = response.quantile_preds[:,median,:20].reshape(-1,3,20).transpose(1,2) + origin[:,None]
    delta = angles(xyz[:,HORIZONS]) - angles(target[:,HORIZONS])
    return torch.linalg.vector_norm(torch.atan2(delta.sin(),delta.cos()),dim=-1).mean()

def train(seed, data, val):
    folder=ROOT/"training/Chronos-2"/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():
        return
    torch.manual_seed(seed)
    model = Chronos2Pipeline.from_pretrained(str(CHECKPOINT),device_map="cuda",torch_dtype=torch.float32).model
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-6)
    start = time.monotonic()
    best, stale = float("inf"), 0
    for epoch in range(1,13):
        model.train()
        batches = torch.randperm(len(data[0]),device="cuda").split(8)
        for step,ids in enumerate(batches,1):
            optimizer.zero_grad(set_to_none=True)
            loss = forward(model,data,ids,True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%10==0:
                write(folder/"progress.txt",f"第 {epoch}/12 轮，第 {step}/{len(batches)} 批，已用 {time.monotonic()-start:.0f} 秒")
        model.eval()
        with torch.no_grad():
            total=sum(float(forward(model,val,ids,False))*len(ids) for ids in torch.arange(len(val[0]),device="cuda").split(16))
        metric=total/len(val[0])
        if not np.isfinite(metric):
            raise ValueError("nonfinite validation metric")
        with (folder/"curve.jsonl").open("a") as f:
            print(json.dumps(dict(epoch=epoch,validation_rad=metric,elapsed_s=time.monotonic()-start)),file=f)
        if metric<best:
            temp=folder/"best.partial"
            torch.save(model.state_dict(),temp)
            temp.replace(folder/"best.pt")
            best,stale=metric,0
        else:
            stale+=1
        if stale>=5 or time.monotonic()-start >= 3600:
            break
    import hashlib
    with (folder/"best.pt").open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    write(folder/"complete.json",json.dumps(dict(seed=seed,validation_rad=best,epochs=epoch,sha256=digest,artifact="center")))
    write(folder/"progress.txt",f"中心训练完成，共 {epoch} 轮，最佳验证误差 {best:.5f} rad")

if __name__=="__main__":
    training,validation=split_data("train"),split_data("validation")
    for seed in SEEDS:
        train(seed,training,validation)
