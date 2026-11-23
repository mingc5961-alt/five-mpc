import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/chen/routeclock_server')
OUT = Path('/data/chen/server_assessment')
PY = str(OUT / 'venv/bin/python')
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TABPFN_DISABLE_TELEMETRY='1')
results = []

def run(name, args, cwd=ROOT, timeout=300):
    start = time.perf_counter()
    with (OUT / f'{name}.log').open('w') as log:
        try:
            r = subprocess.run([PY, *args], cwd=cwd, env=ENV, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    row = dict(name=name, args=args, cwd=str(cwd), exit_code=code, wall_seconds=time.perf_counter()-start)
    results.append(row)
    (OUT / 'checks.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(row), flush=True)

if __name__ == '__main__':
    run('gpu', ['server.py', 'gpu'])
    for project in ['routeclock_five_20261115', 'routeclock_server_full', 'routeclock_platform_20261113', 'routeclock_mpc']:
        run('tests_' + project, ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'], cwd=ROOT/project, timeout=120)
    run('chronos', ['server.py', 'profile', 'chronos', '--repeat', '2', '--out', str(OUT/'profiles')])
    for n in [16, 63]:
        run('tabpfn_' + str(n), ['server.py', 'profile', 'tabpfn', '--features', str(n), '--repeat', '2', '--out', str(OUT/'profiles')])
    run('generate_one', ['server.py', 'five', 'generate', '--split', 'train', '--start', '0', '--stop', '1', '--data', str(OUT/'five_data'), '--out', str(OUT/'five_runs')])
