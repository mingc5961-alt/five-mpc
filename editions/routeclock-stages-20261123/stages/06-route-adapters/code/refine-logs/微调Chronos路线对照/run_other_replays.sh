#!/usr/bin/env bash
set -eu
cd /data/chen/refine-logs/微调Chronos路线对照
source ../专用尺度管实验/env.sh
name="$1"
while [ ! -s "route_heads/heads/$name/special.pkl" ]; do sleep 5; done
for try_idx in 1 2 3; do
    "$PY" -u other_models.py replay "$name" >> "logs/replay-$name.log" 2>&1 && break
    sleep 5
done
