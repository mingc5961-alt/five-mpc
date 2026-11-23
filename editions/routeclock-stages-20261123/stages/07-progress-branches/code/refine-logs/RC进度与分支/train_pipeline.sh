set -eu
cd /data/chen/refine-logs/RC进度与分支
source env.sh
while [ "$(ls packed/validation/*.npz 2>/dev/null|wc -l)" -ne 400 ]; do sleep 3; done
"$PY" learning.py > learning.log 2>&1
"$PY" heads_new.py > heads.log 2>&1
