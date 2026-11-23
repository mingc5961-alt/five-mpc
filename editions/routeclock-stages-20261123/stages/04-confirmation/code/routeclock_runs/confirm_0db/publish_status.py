from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
target = Path("/data/chen/refine-logs/实时实验进度.txt")
temporary = target.with_suffix(".tmp")
while True:
    status = subprocess.run([sys.executable, "-u", str(root / "status.py")], text=True, capture_output=True)
    snapshot = status.stdout if status.returncode == 0 else status.stderr
    translated = {"running": "正在运行", "done": "已完成阶段", "failed": "失败阶段"}
    stages = {"amovfly_index": "AMOVFLY 下载与解压", "amovfly_quality": "AMOVFLY 轨迹质量初筛", "initial_audit": "旧基线初查", "restore_code": "实验代码恢复", "f_0db_pilot": "F 基线 0 dB 试跑"}
    for old, new in {**translated, **stages}.items():
        snapshot = snapshot.replace(old, new)
    initial = "RouteClock 实时实验进度\n预计正式闭环约 2,445 场，最终数量以复用和实筛结果为准。\n"
    temporary.write_text(initial + snapshot, encoding="utf-8")
    temporary.replace(target)
    time.sleep(5)
