#!/usr/bin/env bash
set -u
cd /data/chen/routeclock_runs/confirm_0db || exit 1
exec 9>state/batch_supervisor.lock
flock -n 9 || exit 0
while [[ ! -f state/jobs/summary.done ]]; do
    /data/chen/server_assessment/venv/bin/python -u batch_queue.py
    rc=$?
    if [[ $rc != 0 ]]; then date '+%F %T 调度器退出，将从逐场完成记录恢复'; fi
    [[ -f state/jobs/summary.done ]] || sleep 15
done
