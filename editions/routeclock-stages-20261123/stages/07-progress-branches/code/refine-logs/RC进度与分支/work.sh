set -eu
cd /data/chen/refine-logs/RC进度与分支
source env.sh
case "$1" in
 test)
 while [ "$(ls packed/calibration/*.npz 2>/dev/null|wc -l)" -lt 398 ]; do sleep 4; done
 "$PY" test_data.py > test_data.log 2>&1
 while [ ! -f heads/joint/special.json ]; do sleep 4; done
 "$PY" evaluate.py test > evaluate_test.log 2>&1
 "$PY" evaluate.py world > evaluate_world.log 2>&1
 ;;
 extra)
 while [ ! -f learning.log ]; do sleep 4; done
 "$PY" replay_new.py RC > rc2.log 2>&1 &
 "$PY" replay_new.py Chronos-native Chronos-2 > chronos2.log 2>&1 &
 "$PY" replay_new.py RAFT VECTOR > vector_raft2.log 2>&1 &
 wait
 ;;
 joint)
 while [ ! -f heads/joint/special.json ]; do sleep 4; done
 for i in 1 2 3; do
 "$PY" replay_new.py joint-refit joint-transfer > "joint_$i.log" 2>&1 &
 done
 wait
 ;;
esac
