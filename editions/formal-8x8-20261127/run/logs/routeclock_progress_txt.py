"""Refresh a standalone text snapshot of the formal 8/8 queue every minute."""

import csv
import fcntl
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


RUN = Path("/data/chen/formal_8x8_20261125")
OUTPUT = Path("/data/chen/refine-logs/8x8正式实验实时进度.txt")
PHASES = {
    "R01": "初验参照", "R02": "主线训练与校准", "R03": "10/16 场景对照",
    "R04": "8/8 世界准备", "R05": "共同历史评估", "R06": "门控共同历史评估",
    "R07": "8/8 三档闭环", "R08": "历史窗口与误差管", "R09": "训练配比对照",
    "R10": "统计汇总",
}
PURPOSE = {
    "R01": "登记旧版 10/16 初验的风险和吞吐差值，保留比较起点。本轮服务器实测值由后续阶段产生。",
    "R02": "分别训练读取 8 秒或 180 秒历史的 Transformer 和 GRU。每款用 3 个随机初值，在验证集选权重，再拟合路线融合权重、准备各方法的误差管，供后续预测和闭环调用。",
    "R03": "检验随机机动的飞机增多后，主线结论是否仍成立。沿用 8/8 场景的种子和飞机排列，把两架稳定重访机换成随机机；在紧档预算下重测预测误差和通信表现。",
    "R04": "生成主实验共用的 13 个模拟世界，每个世界含 16 架飞机，8 架有可复用路线、8 架随机机动。世界 0–2 用于开发选设置，100–109 用于独立确认。",
    "R05": "单独比较预测能力。让各方法看到同一组观测，每架飞机每 3 秒观测一次，每世界共 960 个预测起点；用模拟真值计算未来 1、2、5、10 秒的指向角误差。",
    "R06": "检验模型是否知道什么时候该信历史路线。对短期 Transformer 和 GRU 分别比较只用短期预测、只用路线、各占一半、按预测时长固定加权、随新观测动态加权；分重访、偏离和随机机动查看角误差。",
    "R07": "主通信实验。在同一个世界，让各方法自行决定何时观测、如何指向飞机，运行 180 秒服务。14 种方法共 2,394 场，紧档完整扫描。中松档 14 种方法在 10 个确认世界比较原倍率，六种辅助方法仅在 3 个确认世界补全其余倍率；四种核心方法保留完整扫描，逐场记录净吞吐、风险和实际感知时间。",
    "R08": "检验更早的观测能否帮助预测。四种方法在同一组付费观测上比较 30、60、120、180 秒；30、120、180 秒在紧档、1.00 倍率下比较闭环。开发世界 0，确认世界 100–104。另测门控只缩短路线记忆，以及门控和微调 Chronos 沿用 180 秒误差管。",
    "R09": "检验训练时多见随机机动轨迹能否改善泛化。将每 16 架中 2 架随机和 8 架随机两种配比分别用于训练、验证和校准，再在同一批独立 8/8 世界评估。开发阶段按风险不超过 5% 选择 GRU 门控配置。",
    "R10": "把同一世界里的方法逐一配对，比较相近风险下的净吞吐，也报告开发集选定策略在确认集的表现。汇总四个预测时长的角误差、感知开销及相对微调 Chronos-2 的差值和 95% 区间。",
}
MODELS = {
    "C": "纯路线记忆（只取历史路线预测中心）",
    "T8": "短期 Transformer（8 秒历史，修正 IMM 预测）",
    "T180": "长期 Transformer（180 秒输入窗，修正 IMM 预测）",
    "G8": "短期 GRU（8 秒历史，修正 IMM 预测）",
    "G180": "长期 GRU（180 秒输入窗，修正 IMM 预测）",
    "T8P": "短期 Transformer 与历史路线动态加权",
    "G8P": "短期 GRU 与历史路线动态加权",
    "Cnative": "微调 Chronos-2，使用原生误差管",
    "Czero": "公开基础版 Chronos-2，未经本任务微调",
    "Cexternal": "微调 Chronos-2，使用外接 LightGBM 误差管",
    "IMM": "交互多模型运动滤波器（按运动状态外推）",
    "VECTOR": "VECTOR（从预测速度推算位移）",
    "RAFT": "RAFT（从冻结的轨迹库中检索）",
    "Joint07": "阶段 07 RouteClock（联合修正路线进度和分支）",
}
BUDGETS = {"T": "紧档，最多 3 秒", "M": "中档，最多 4.5 秒", "L": "松档，最多 9 秒"}
VERSIONS = {"arch": "主线配置", "original": "2/16 随机训练配比", "half": "8/16 随机训练配比"}
KINDS = {
    "anchor": "登记旧结果", "world": "生成世界", "shard": "整理样本", "train": "训练模型",
    "gate": "拟合门控", "cal": "校准误差管", "panel": "共同观测预测评估",
    "cell": "通信闭环", "tables": "汇总统计",
}


def world_label(w):
    if w < 1000 or w >= 3000:
        number = w % 3000
        scene = "10/16" if w >= 3000 else "8/8"
        use = "开发" if number < 100 else "确认"
        return f"{scene} {use}世界 {w}"
    offset = w % 1000
    use = next(name for limit, name in [(12, "训练"), (16, "验证"), (20, "误差管拟合"), (28, "校准")] if offset < limit)
    return f"{'8/16' if w < 2000 else '2/16'} 随机配比·{use}世界 {w}"


def table_label(code):
    fixed = {"all_main": "主场景全部14种配置", "core_main": "主场景门控、Chronos、IMM、纯路线四组", "core_history": "四种方法的历史窗口、分组角误差和误差管配对比较", "main_T": "主场景紧档", "budgets_ML": "主场景中档与松档", "random10": "10/16 随机场景",
             "training_half": "8/16 随机训练配比", "training_original": "2/16 随机训练配比"}
    if code in fixed:
        return fixed[code]
    window, variant = code.split("_", 1)
    variants = {"priority": "四个优先模型", "all": "全部模型", "route": "只缩短路线记忆",
                "tube_transfer": "沿用 180 秒误差管"}
    return f"{window[1:]} 秒历史，{variants[variant]}"


def describe(task):
    kind, args = task["kind"], task["args"]
    if kind == "tables":
        return f"汇总{table_label(args['code'])}的逐世界角误差、风险和吞吐结果。"
    if kind == "anchor":
        return "登记旧版 10/16 初验的参照数值及来源。"
    if kind == "world":
        return f"生成{world_label(args['w'])}的 16 架飞机轨迹，并保存偏离事件和世界指纹。"
    version = VERSIONS[args["v"]]
    if kind == "shard":
        split = "训练" if args["split"] == "fit" else "验证"
        return f"为{version}整理{split}样本，保存观测历史、运动滤波预测和真实轨迹。"
    model = f"{MODELS[args['a']]} [{args['a']}]"
    if kind == "train":
        return f"用{version}训练{model}，比较三个随机初值，按验证误差保存权重。"
    if kind == "gate":
        return f"为{model}拟合各预测时长的融合权重，学习如何依据已观测到的误差调整对路线的信任（{version}）。"
    history = (f"仅路线记忆截为 {args['H']} 秒" if args["mode"] == "route"
               else f"可用历史上限 {args['H']} 秒")
    config = f"{history}，{version}"
    if kind == "cal":
        return f"为{model}准备误差管，在校准样本上核对各预测时长的实际覆盖率（{config}）。"
    where = world_label(args["w"])
    if kind == "panel":
        return f"{where}，用同一组观测评估{model}未来 1、2、5、10 秒的角误差（{config}）。"
    tube = "，沿用 180 秒校准管" if args["transfer"] else ""
    return (f"{where}，{model}。运行 180 秒通信闭环，测净吞吐、风险和感知时间。"
            f"感知预算为{BUDGETS[args['b']]}。误差管 ×{args['m']:g}{tube}，{config}。")


def read_snapshot():
    # The queue publishes JSON just before its matching CSV.
    for _ in range(5):
        status = json.loads((RUN / "live/status.json").read_bytes())
        with (RUN / "live/tasks.csv").open(encoding="utf-8-sig", newline="") as src:
            rows = list(csv.DictReader(src))
        if Counter(row["状态"] for row in rows) == status["counts"] and (
            {row["任务ID"] for row in rows if row["状态"] == "running"}
            == {job[0] for job in status["running"]}
        ):
            return status, rows
        time.sleep(0.2)
    raise ValueError("状态与逐任务明细正在切换")


def write_snapshot(lines):
    temporary = OUTPUT.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)


def publish(tasks):
    status, rows = read_snapshot()
    counts = Counter(status["counts"])
    total = status["total"]
    cells = [r for r in rows if r["任务ID"].startswith("cell__")]
    done_cells = sum(r["状态"] == "done" for r in cells)
    now = datetime.now().astimezone()
    heartbeat = datetime.fromisoformat(status["updated"])
    ended = counts["done"] + counts["failed"] + counts["blocked"] == total
    lines = [
        "RouteClock　8/8 正式实验实时进度",
        f"更新于 {now:%Y-%m-%d %H:%M:%S %Z}　每 60 秒刷新",
        f"调度心跳　{heartbeat:%Y-%m-%d %H:%M:%S}　调度 PID {status['pid']}",
        f"总任务　{counts['done']}/{total}（{counts['done']/total:.1%}）　"
        f"正式闭环　{done_cells}/{len(cells)} 场（{done_cells/len(cells):.1%}）",
        f"队列　运行 {counts['running']}　就绪 {counts['ready']}　等前置 {counts['waiting']}　"
        f"失败 {counts['failed']}　前置失败 {counts['blocked']}",
    ]
    if not ended and (now - heartbeat).total_seconds() > 60:
        lines.append("调度心跳已超过一分钟，上述数字为最近一次发布的状态。")
    lines.append(f"资源　内存 {status['memory_GiB']:.1f} GiB　磁盘可用 {status['free_GiB']:.1f} GiB")
    lines.extend(["", "这轮实验要回答什么",
                  "  16 架飞机共享一个基站，观测飞机位置会占用通信时间。主要比较短期 GRU 加历史路线与微调 Chronos-2，看哪种方法在相同感知预算、相近风险下指得更准、实际传得更多。",
                  "  一个世界是一组可重复模拟的飞行轨迹和无线环境。开发世界用来选设置，确认世界用来检验选好的设置。每场闭环让模型作出指向和观测决定，这些决定会影响下一步能看到什么。",
                  "  误差管表示预测方向周围预留的角度范围。倍率 1 使用校准后的原宽度，0.75 表示取其 75%。净吞吐以 Mbps 计，风险包括没有可用动作或波束指向失准。",
                  "", "此刻正在做什么（耗时为本次任务已运行的墙钟时间）"])
    for key, phase, resource, _, attempt, seconds, _ in status["running"]:
        lines.extend([f"  {phase} · {KINDS[tasks[key]['kind']]} · {resource.upper()} · 已运行 {seconds:.0f} 秒 · 第 {attempt} 次",
                      "    " + describe(tasks[key]), "    任务 ID　" + key])
    if not status["running"]:
        lines.append("  当前无运行任务。")
    lines.extend(["", "各阶段安排与进度（完成数 / 计划数）",
                  "  就绪表示前置任务已齐、等待调度；等前置表示依赖的世界、权重或结果尚未齐备。不同阶段可同时运行。"])
    breakdown = Counter((r["批次"], tasks[r["任务ID"]]["kind"]) for r in rows)
    finished = Counter((r["批次"], tasks[r["任务ID"]]["kind"]) for r in rows if r["状态"] == "done")
    due = {"running": "运行", "ready": "就绪", "waiting": "等前置", "failed": "失败", "blocked": "前置失败"}
    for phase, values in sorted(status["phases"].items()):
        state = "，".join(f"{label} {values[k]}" for k,label in due.items() if values.get(k)) or "全部完成"
        parts = [f"{label} {finished[phase, k]}/{breakdown[phase, k]}"
                 for k, label in KINDS.items() if breakdown[phase, k]]
        lines.extend(["", f"  {phase}　{PHASES[phase]}　{values.get('done', 0)}/{sum(values.values())}　{state}",
                      "    " + PURPOSE[phase], "    分项　" + "，".join(parts)])
    lines.extend(["", "14 种模型配置速查", *(f"  {code}　{name}" for code, name in MODELS.items())])
    if ended:
        lines.append("本轮队列已结束，分钟记录结束。")
    write_snapshot(lines)
    return ended


def main():
    with open(__file__, "rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        manifest = RUN / "manifest.json"
        version = None
        while True:
            tick = time.monotonic()
            try:
                stamp = manifest.stat().st_mtime_ns
                if stamp != version:
                    tasks = {task["id"]: task for task in json.loads(manifest.read_bytes())}
                    version = stamp
                if publish(tasks):
                    return
            except (OSError, ValueError, KeyError) as error:
                print(datetime.now().isoformat(), error, file=sys.stderr, flush=True)
            time.sleep(max(0, 60 - (time.monotonic() - tick)))


if __name__ == "__main__":
    main()
