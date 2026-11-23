set -eu
cd /data/chen/refine-logs/RC进度与分支
source env.sh
"$PY" flights_local.py > flight_prep.log 2>&1
"$PY" evaluate.py flight > evaluate_flight.log 2>&1
