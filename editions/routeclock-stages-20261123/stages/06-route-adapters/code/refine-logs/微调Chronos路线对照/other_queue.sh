#!/usr/bin/env bash
set -eu
cd /data/chen/refine-logs/微调Chronos路线对照
source ../专用尺度管实验/env.sh
for name in VECTOR RAFT; do
    while [ ! -s "${name}_adapter.pt" ]; do sleep 5; done
    "$PY" -u other_models.py tube "$name" > "logs/tube-$name.log" 2>&1
    "$PY" -u other_models.py centers "$name" > "logs/centers-$name.log" 2>&1
done
