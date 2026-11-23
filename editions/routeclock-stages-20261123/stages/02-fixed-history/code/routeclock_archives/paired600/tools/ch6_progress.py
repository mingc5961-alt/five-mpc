"""Publish a live Markdown view of the running paired-world experiment."""
import fcntl
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1] / "runs" / "paired600"
PAGE = ROOT / "实验进度.md"
TZ = ZoneInfo("Asia/Shanghai")
STAGES = {"validation": "验证与固定基线择优", "confirmation": "独立测试", "complete": "已完成"}


def timestamp(seconds):
    return datetime.fromtimestamp(seconds, TZ).strftime("%Y-%m-%d %H:%M:%S")


def supervisor_running():
    with (ROOT / "supervisor.lock").open() as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def block(group, job):
    if group == "fixed": return "固定基线择优"
    if group == "validation": return "网络验证"
    if job["scheduler"] != "value": return "调度策略"
    if job["method"] in ("C", "CR_raw"): return "Chronos 对照"
    if job["cost_ms"] != 3.1: return "感知成本压力"
    return "共享预算主比较"


def publish(plan):
    status = json.loads((ROOT / "status.json").read_text())
    now = time.time()
    stage = status["stage"]
    live = supervisor_running()
    active = set(status.get("running", [])) if live and now-status["at"] < 120 else set()
    total, done, running = Counter(), Counter(), Counter()
    test_total, test_done = Counter(), Counter()
    latest = []
    for group, jobs in plan.items():
        for job in jobs:
            name = block(group, job)
            total[name] += 1
            out = Path(job["out"])
            marker = out / "completed.json"
            if group == "confirmation": test_total[job["world"]] += 1
            if marker.is_file():
                done[name] += 1
                if group == "confirmation": test_done[job["world"]] += 1
                latest.append((marker.stat().st_mtime, out, job))
            elif str(out) in active:
                running[name] += 1
    passed = sum(done.values())
    worlds = sum(test_done[i] == n for i, n in test_total.items())
    health = STAGES[stage]
    if stage != "complete":
        if not live: health += " · 调度进程已退出"
        elif now-status["at"] >= 120: health += " · 等待调度状态更新"
        else: health += " · 运行中"
    failures = status.get("failed", [])
    lines = ["# RouteClock 实验进度", "", f"**{health}**　｜　北京时间 {timestamp(now)}",
             "", f"已通过审计 **{passed} / {sum(total.values())}** 项　｜　当前并发 **{sum(running.values())}** 项　｜　当前阶段已报失败 **{len(failures)}** 项",
             "", "| 实验 | 已完成 | 运行中 | 计划 | 完成比例 |",
             "|---|---:|---:|---:|---:|"]
    hw=status.get('resources',{})
    if 'gpu_pct' in hw and stage!='complete':
        lines[4:4]=[f"GPU 利用率 **{hw['gpu_pct']}%**（最近至多 6 次采样均值 {hw['gpu_avg_pct']}%），"
                     f"显存 **{hw['gpu_mib']} / {hw['gpu_total_mib']} MiB**，"
                     f"已派发 GPU 任务 **{hw['gpu_jobs']} / {hw['gpu_limit']}**。"
                     f"CPU 实测 **{hw['cpu_cores']:.1f} 核**，整机并发上限 {hw['worker_limit']} 项。",""]
    for name in ("网络验证", "固定基线择优", "共享预算主比较", "感知成本压力", "Chronos 对照", "调度策略"):
        lines.append(f"| {name} | {done[name]} | {running[name]} | {total[name]} | {done[name]/total[name]:.1%} |")
    lines += ["", f"**独立测试共完成 {sum(test_done.values())} / {sum(test_total.values())} 场**，其中 {worlds} / {len(test_total)} 个世界已跑齐预定策略。每组固定基线择优包含 20 个验证世界，整组完成后计 1 项。",
              "", "验证与择优全部通过后，程序冻结模型并启动独立测试。每 15 秒更新此页，以通过审计的完成标记计数。",
              "", "## 最近完成", ""]
    for t, out, job in sorted(latest, key=lambda x: x[0], reverse=True)[:5]:
        if job["method"] == "fixed-grid":
            label, artifact = f"固定基线 · 预算 {job['budget_pct']:.3g}%", "winner.json"
        else:
            label = (f"{'测试' if job['split']=='test' else '验证'}世界 {job['world']} · {job['method']} · "
                     f"{job['cost_ms']:g} ms / {job['budget_pct']:.3g}% · {job['warm']} · "
                     f"{job['model']}/{job['calibration']} · {job['scheduler']}")
            artifact = "metrics.json"
        lines.append(f"- {timestamp(t)}　[{label}]({(out/artifact).relative_to(ROOT)})")
    if not latest: lines.append("首批任务正在生成结果。")
    if failures:
        lines += ["", "## 已报失败", ""] + [f"- {f}" for f in failures]
    lines += ["", "[实验说明](实验说明.md) · [600 场测试清单](独立测试600场.csv) · [调度状态](status.json) · [调度日志](logs/supervisor.log)"]
    if (ROOT/"确认结果.md").is_file(): lines += ["", "[独立测试结果](确认结果.md)"]
    tmp = PAGE.with_suffix(".tmp")
    tmp.write_text("\n".join(lines)+"\n", encoding="utf-8")
    tmp.replace(PAGE)
    return stage


def main():
    with (ROOT/"progress.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        plan = json.loads((ROOT/"manifest.json").read_text())
        while publish(plan) != "complete":
            time.sleep(15)


if __name__ == "__main__": main()
