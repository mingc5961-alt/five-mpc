#!/usr/bin/env python3

from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parent
cgroup = Path("/sys/fs/cgroup")

def cpu():
    return int(dict(line.split() for line in (cgroup / "cpu.stat").read_text().splitlines())["usage_usec"])

quota, period = map(int, (cgroup / "cpu.max").read_text().split())
cpu_limit = quota / period
started = int((root / "state/start_epoch").read_text())
first_cpu = cpu()
first_time = time.monotonic()
time.sleep(0.3)
used_cpu = (cpu() - first_cpu) / (time.monotonic() - first_time) / 1e6
memory = int((cgroup / "memory.current").read_text())
memory_limit = int((cgroup / "memory.max").read_text())
print(time.strftime("%F %T"), f"本批耗时 {int(time.time()-started)//60}m {int(time.time()-started)%60}s")
print(f"CPU {used_cpu:.2f}/{cpu_limit:g} 核  内存 {memory/2**30:.1f}/{memory_limit/2**30:.1f} GiB  磁盘剩余 {shutil.disk_usage(root).free/2**30:.1f} GiB")
subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], check=False)
states = sorted((root / "state").glob("*.*"))
for status in ("running", "queued", "done", "failed"):
    names = [p.stem for p in states if p.suffix == f".{status}"]
    print(f"{status:7} {len(names):3}  {', '.join(names)}")
print("正式闭环结果", len(list((root / "results").glob("**/complete.json"))))
print("日志目录", root / "logs")

def completed(stage):
    return sum(1 for item in (root / "results" / stage).glob("**/complete.json") if item.is_file())

def line(label, done, total, unit, suffix=""):
    print(f"{label}  {done}/{total} {unit}  {100 * done / total:.1f}%{suffix}")

print("\n各阶段完成比例")
line("资产准备", sum((root / "state" / f"{name}.done").exists() for name in ("restore_code", "amovfly_index", "initial_audit", "amovfly_quality")), 4, "项", "（AMOVFLY 最终选窗另计）")
line("0 dB 基线试跑", sum((root / "state" / f"{name}_0db_pilot.done").exists() for name in ("i", "f", "rc")), 3, "场")
for model in ("Chronos-2", "VECTOR", "RAFT"):
    count = sum((root / "training" / model / str(seed) / "complete.json").is_file() for seed in (7, 11, 19))
    line(f"{model} 中心权重", count, 3, "个")
    for seed in (7, 11, 19):
        file = root / "training" / model / str(seed) / "progress.txt"
        if file.is_file():
            print(f"  种子 {seed}  {file.read_text().strip()}")
line("误差管开发", completed("development"), 180, "场", "（上限）")
line("先进模型确认", completed("confirmation"), 1080, "场")
line("I/F/RC 基线补跑", completed("baselines"), 360, "场", "（本批新世界清单）")
line("AMOVFLY C3 单机回放", completed("amovfly"), 825, "场", "（按 55 架次估算）")

print("\n正在运行和等待中的实验")
def live_workers(script):
    found = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            args = (process/"cmdline").read_bytes().split(b"\0")
            if process.joinpath("cwd").resolve() == root and len(args)>1 and Path(args[1].decode()).name == script:
                found.append(int(process.name))
        except (OSError, UnicodeError):
            continue
    return found

baseline_pids = live_workers("run_baselines.py")
baseline_running = bool(baseline_pids)
baseline_done = completed("baselines")
if baseline_running:
    print(f"正在运行：I/F/RC 正式基线，0 dB，按世界并行，检测到 {max(0,len(baseline_pids)-1)} 个子进程。已完成 {baseline_done}/360 场。")
    print(f"已提交队列：40 个世界，每世界 9 场。还有 {360-baseline_done} 场未完成，包含正在执行的场次。")
else:
    print("正式基线当前未运行。")
for method in ("I", "F", "RC"):
    count = sum(1 for path in (root / "results" / "baselines").glob(f"*_{method}/complete.json") if path.is_file())
    print(f"  {method}  {count}/120 场")
log = root / "logs" / "formal_baselines.log"
if log.is_file():
    for row in log.read_text().splitlines()[-3:]:
        values = row.split()
        if len(values) == 4 and values[0].isdigit():
            world, budget, method, elapsed = values
            print(f"  最近完成：世界 {world}，预算 {budget}，方法 {method}，{elapsed} 秒")

active_cells = sorted(d.name for d in (root/"results/baselines").glob("*") if d.is_dir() and not (d/"complete.json").exists())
if baseline_running:
    print("已进入且尚未写完的格子（世界_预算_方法）", ", ".join(active_cells) or "等待分配")
if live_workers("train_vector_long.py"):
    print("正在执行：VECTOR 长窗 GPU 训练，种子按 7、11、19 顺序执行。")
elif (root/"state/train_VECTOR.failed").exists():
    print("VECTOR 上一次训练失败，见 logs/train_VECTOR.log")
if live_workers("train_chronos_long.py"):
    print("正在执行：Chronos-2 全参数微调，种子按 7、11、19 顺序执行。")
elif (root/"state/train_Chronos-2.queued").exists():
    print("已排队：Chronos-2 全参数微调，VECTOR 释放 GPU 后自动启动。")
elif (root/"state/train_Chronos-2.failed").exists():
    print("Chronos-2 启动或训练失败，详见 logs/train_Chronos-2.log。")
if live_workers("train_raft_long.py"):
    print("正在执行：RAFT 联合 XYZ 检索与 GPU 训练。")
elif (root/"state/train_RAFT.done").exists():
    print("RAFT 三个种子中心权重已完成。")
elif (root/"state/train_RAFT.failed").exists():
    print("RAFT 训练失败，见 logs/train_RAFT.log。")
advanced_pids = live_workers("run_advanced.py")
print(f"先进模型队列：检测到 {max(0,len(advanced_pids)-1)} 个工作进程。")
for model in ("Chronos-2","VECTOR","RAFT"):
    folder = root/"tubes"/model
    fitted = (folder/"7"/"offline.json").is_file()
    picked = (folder/"selection.json").is_file()
    dev = len(list((root/"results/development"/model).glob("**/complete.json")))
    confirm = len(list((root/"results/confirmation"/model).glob("**/complete.json")))
    print(f"  {model}  种子7离线校准{'完成' if fitted else '尚未落盘'}；开发 {dev}/60 场；选型{'已冻结' if picked else '待定'}；确认 {confirm}/360 场")
print("每个模型开发完即选型并确认；AMOVFLY C3 另需冻结实飞窗口。")
