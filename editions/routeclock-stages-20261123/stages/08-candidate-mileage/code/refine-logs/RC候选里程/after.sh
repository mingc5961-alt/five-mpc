set -eu
cd /data/chen/refine-logs/RC候选里程
source env.sh
while [ ! -s selection.json ]; do sleep 3; done
"$PY" tube.py > tube.log 2>&1
for i in 1 2 3 4; do "$PY" matrix.py v2-transfer v2-refit > "v2_$i.log" 2>&1 & done
(
 for s in test prior flight fresh; do "$PY" score.py "$s" > "score_$s.log" 2>&1; done
) &
wait
