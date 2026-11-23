set -eu
cd /data/chen/refine-logs/RC进度与分支
source env.sh
while [ "$(ls cells/*.json|wc -l)" -lt 210 ] || [ ! -f world_oracle.csv ] || [ ! -f test_oracle.csv ]; do sleep 5; done
"$PY" timings.py > timings.log 2>&1
"$PY" summarize_new.py > summary.log 2>&1
