"""独立运行目录、配置检查、数据生成与可复查进度。"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import pickle
import os
import tempfile
import numpy as np
from .data import make_scene, load_config

def atomic(path,writer):
    """大文件写完后原子替换，重跑时只认完整落盘的文件。"""
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        tmp=Path(f.name)
    try:
        with tmp.open("wb") as f:
            writer(f)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)

def save_json(path, data):
    raw=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False).encode("utf8")
    atomic(path,lambda f:f.write(raw))

def log(out,stage,message):
    row={"time":datetime.now().isoformat(timespec="seconds"),"stage":stage,"message":message}
    print(f"[{row['time']}] {stage}  {message}",flush=True)
    with (out/"progress.jsonl").open("a",encoding="utf8") as f:
        f.write(json.dumps(row,ensure_ascii=False)+"\n")
    save_json(out/"status.json",row)

def validate(c):
    if any(type(c[k])!=int or c[k]<1 for k in ("uavs","training_uavs","steps","slow_every","slow_horizon","fast_horizon","search_width")):
        raise ValueError("节点数、步数和控制时域必须为正整数。")
    if not (c["warmup_s"]>=80 and c["warmup_s"]%2==0 and c["slow_horizon"]>=c["slow_every"]):
        raise ValueError("预热应为不小于 80 的偶数，慢层展望需覆盖整个执行帧。")
    if not (0<c["min_airtime_fraction"]<=1 and 0<c["sensing_budget"]<=1 and
            c["target_network_mbps"]>=0 and c["deficit_weight"]>=0):
        raise ValueError("资源比例、最低服务目标或欠账权重无效。")
    sim=load_config()["simulation"]
    if c["dt"]<=sim["observation_cost_s"]+sim["beam_switch_s"]*c["uavs"]:
        raise ValueError("快轮时长不足以承载一次观测和各机切束。")
    if set(c["policies"])-{"rotation","joint_mpc","age_mpc","myopic"} or not c["policies"] or len(c["policies"])!=len(set(c["policies"])):
        raise ValueError("策略列表为空、重复或含未知策略。")
    for key in ("gate_ids","train_ids","tube_cal_ids","cal_ids","test_ids"):
        ids=c[key]
        if not ids or len(set(ids))!=len(ids) or any(type(v)!=int or v<0 for v in ids):
            raise ValueError(f"{key} 需要互异的非负整数。")
    alpha=c["tube_alpha"]
    if not 0<alpha<1 or np.ceil((len(c["tube_cal_ids"])+1)*(1-alpha))>len(c["tube_cal_ids"]):
        raise ValueError("独立误差管校准网络数量不足以估计所选覆盖分位数。")

def build_one(job):
    split,i,c,out=job
    path=out/f"{split}_{i}.pkl"
    if path.exists():
        return path.name+" 已就绪"
    data=make_scene(split,i,c)
    atomic(path,lambda f:pickle.dump(data,f,protocol=5))
    return path.name+" 已生成"

def prepare(c,out,workers,splits=("gate","train","cal","test","tube_cal")):
    jobs=[(s,i,c,out) for s in splits for i in c[s+"_ids"] if not (out/f"{s}_{i}.pkl").exists()]
    if not jobs:
        return
    log(out,"准备场景",f"{len(jobs)} 个网络，{workers} 个本地进程")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(build_one,j) for j in jobs]
        for k,f in enumerate(as_completed(pending),1):
            log(out,"准备场景",f"{k}/{len(jobs)} {f.result()}")

def load_split(out,c,split):
    samples=[]
    for i in c[split+"_ids"]:
        with (out/f"{split}_{i}.pkl").open("rb") as f:
            samples.extend(pickle.load(f))
    return samples
