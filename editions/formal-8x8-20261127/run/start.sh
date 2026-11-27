#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source env.sh
exec "$PY" -u supervisor.py
