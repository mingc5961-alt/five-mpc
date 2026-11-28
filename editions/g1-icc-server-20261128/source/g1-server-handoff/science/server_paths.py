"""Paths to the pinned protocol, existing RouteClock physics and isolated outputs."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('G1_STAGE04', '/data/chen/routeclock_runs/confirm_0db'))
ENGINE = BASE / 'work/code/routeclock_refine'
MEMORY = Path('/data/chen/refine-logs/微调Chronos路线对照')
UPSTREAM = ROOT / 'reference/five-mpc/editions/g1-paid-cold-start-20261127/evidence/work'
RUN = Path(os.environ.get('G1_RUN', ROOT/'runs/rebuild_v1')).resolve()
for directory in (ENGINE, BASE, MEMORY, UPSTREAM):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
