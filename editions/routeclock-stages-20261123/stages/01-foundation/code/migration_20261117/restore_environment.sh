#!/usr/bin/env bash
set -euo pipefail
cd /data/chen
kit=/data/chen/migration_20261117
base_python=${BASE_PYTHON:-python3.12}
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
target=/data/chen/server_assessment/venv
if [[ -e "$target" ]]; then
  echo "Existing environment at $target. Use it if verified, or move it aside before rebuilding."
  exit 1
fi
"$base_python" -m venv "$target"
"$target/bin/python" -m pip install 'torch==2.10.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --constraint "$kit/requirements-constraints.txt"
"$target/bin/python" -m pip install \
  --requirement /data/chen/routeclock_server/requirements-linux.txt \
  --constraint "$kit/requirements-constraints.txt"
"$target/bin/python" -m pip check
"$target/bin/python" "$kit/resume_experiments.py" --check-only
echo 'Environment ready. Start: /data/chen/server_assessment/venv/bin/python /data/chen/migration_20261117/resume_experiments.py'
