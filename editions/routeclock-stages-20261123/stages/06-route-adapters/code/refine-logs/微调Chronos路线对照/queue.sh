#!/usr/bin/env bash
set -eu
cd /data/chen/refine-logs/微调Chronos路线对照
source ../专用尺度管实验/env.sh
mkdir -p logs cells
pids=()
for idx in 1 2 3 4 5 6; do
    "$PY" -u replay_route.py gpu > "logs/gpu-$idx.log" 2>&1 &
    pids+=("$!")
done
"$PY" -u replay_route.py cpu > logs/cpu.log 2>&1 &
pids+=("$!")
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
