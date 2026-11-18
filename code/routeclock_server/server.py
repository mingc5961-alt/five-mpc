"""RouteClock Linux 包入口：在各自的工作目录启动原程序，共用本包离线权重。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIVE = ROOT / "routeclock_five_20261115"
MPC = ROOT / "routeclock_server_full"
HUB = ROOT / "weights/huggingface/hub/models--amazon--chronos-2"
TAB = ROOT / "weights/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "gpu", "profile", "five", "mpc", "paired"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.command == "verify":
        files = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["sha256"]
        for name, expected in files.items():
            with (ROOT / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise SystemExit(f"校验失败: {name}")
        print(f"OK: {len(files)} files")
        return
    if options.command == "gpu":
        import torch
        print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, flush=True)
        if not torch.cuda.is_available():
            raise SystemExit("当前 Python 无可用 CUDA。")
        free, total = torch.cuda.mem_get_info()
        print(torch.cuda.get_device_name(), f"free {free/2**30:.2f} / total {total/2**30:.2f} GiB")
        return
    revision = (HUB / "refs/main").read_text().strip()
    checkpoint = HUB / "snapshots" / revision
    env = os.environ.copy()
    env.update(HF_HOME=str(ROOT/"weights/huggingface"), HF_HUB_OFFLINE="1",
               CHRONOS_DIR=str(checkpoint), ROUTECLOCK_TABPFN_CHECKPOINT=str(TAB))
    args = options.args
    if options.command == "mpc":
        directory, cmd = MPC, ["run.py", *args]
    elif options.command == "paired":
        if not args or args[0] not in ("run","audit","search","summary"):
            parser.error("paired 子命令可用 run / audit / search / summary")
        module = {"run":"paired_refresh", "audit":"paired_refresh_audit",
                  "search":"paired_search_trace", "summary":"paired_refresh_summary"}[args[0]]
        directory, cmd = MPC, ["-m", f"mpc.{module}", *args[1:]]
    else:
        directory = FIVE
        script = "day1_profile" if options.command == "profile" else "five_campaign"
        cmd = [f"tools/{script}.py", *args]
        if args and args[0] in ("chronos","timing","network","generate") and "--checkpoint" not in args:
            cmd += ["--checkpoint", str(checkpoint)]
    raise SystemExit(subprocess.call([sys.executable,"-u",*cmd], cwd=directory, env=env))


if __name__ == "__main__":
    main()
