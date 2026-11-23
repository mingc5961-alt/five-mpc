#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ../专用尺度管实验/env.sh
for head in original special; do
 for model in RC Chronos-2 VECTOR RAFT; do
   "$PY" -u bench.py "$model" "$head" > "$model-$head.log" 2>&1
   date "+%F %T　实时性补测 $model $head 完成，16 用户、单事件推理，每 62 ms 固定到达，丢弃 64 个热身事件，保留 1024 个含排队的时延。" >> ../实时实验进度.txt
 done
done
