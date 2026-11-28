"""Shared CPU allocations for the paired simulation and model workers."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
SLOTS=ROOT/'paired_linux_v2/slots'

@contextmanager
def compute_slot():
    SLOTS.mkdir(parents=True,exist_ok=True)
    while True:
        active_old=sum(1 for _ in (ROOT/'recovery_01/historical_diagnostics/active').glob('*.json'))
        max_workers=max(1,46-active_old)
        if int(Path('/sys/fs/cgroup/memory.current').read_text())<84*(2**30) and shutil.disk_usage(ROOT).free>20*(2**30):
            for n in range(max_workers):
                f=(SLOTS/f'{n:02d}.lock').open('a+')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:f.close();continue
                try:
                    f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush()
                    yield
                finally:f.close()
                return
        time.sleep(.5)
