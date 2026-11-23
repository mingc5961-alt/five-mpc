"""Schedule the additional 12.4 ms pressure point alongside the original campaign."""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
import fcntl
import json
import os
import shutil
import time

import ch6_run as runner
from ch6_dispatch import Admission

PRIMARY = runner.ROOT
ROOT = Path('/data/chen/routeclock_runs/paired600_pressure12_4')
COSTS = (3.1, 6.2, 9.3, 12.4)
WORLD_PAIRS = {(w, m) for w in range(30) for m in ('I', 'RC')}


def read(path):
    return json.loads(path.read_text())


def pairs(rows):
    assert len(rows) == 60 and {(r['world'], r['method']) for r in rows} == WORLD_PAIRS
    return rows


def pressure(rows, cost):
    return pairs([r for r in rows if r['cost_ms'] == cost and r['budget_pct'] == 5
                  and r['split'] == 'test' and r['warm'] == 'lifecycle' and r['model'] == 'frozen'
                  and r['calibration'] == 'deadline' and r['scheduler'] == 'value' and r['method'] in ('I', 'RC')])


def register(primary_rows):
    frozen = ROOT / 'models/frozen'
    frozen.mkdir(parents=True, exist_ok=True)
    for name in ('selection.json', 'bundle.pkl'):
        src, dest = PRIMARY / 'models/frozen' / name, frozen / name
        if not dest.exists():
            temp = dest.with_suffix('.tmp')
            shutil.copyfile(src, temp)
            temp.replace(dest)
        assert runner.digest(src) == runner.digest(dest)
    model_hash = runner.digest(frozen / 'bundle.pkl')
    assert model_hash == read(frozen / 'selection.json')['sha256']
    rows = []
    for base in pressure(primary_rows, 9.3):
        out = ROOT / 'test/cost12.4_budget5' / Path(base['out']).name
        argv = base['argv'].copy()
        for key, value in [('--cost-ms', '12.4'), ('--bundle', str(frozen / 'bundle.pkl')), ('--out', str(out))]:
            argv[argv.index(key) + 1] = value
        rows.append(dict(base, cost_ms=12.4, out=str(out), argv=argv))
    plan = dict(confirmation=rows)
    manifest = ROOT / 'manifest.json'
    if manifest.exists():
        assert read(manifest) == plan
    else:
        runner.write(manifest, plan)
    amendment = ROOT / 'amendment.json'
    hashes = dict(primary_manifest_sha256=runner.digest(PRIMARY / 'manifest.json'),
                  supplement_manifest_sha256=runner.digest(manifest), frozen_sha256=model_hash)
    if not amendment.exists():
        status = read(PRIMARY / 'status.json')
        runner.write(amendment, dict(**hashes, registered_at=datetime.now().astimezone().isoformat(),
            primary_completed=status.get('confirmation', status.get('completed')), primary_stage=status['stage'],
            analysis='Compare RC minus I net Mbps by matched world at four costs. Estimate the change from 9.3 to 12.4 ms with a paired bootstrap, 2000 replicates. Supplementary analysis.'))
    assert all(read(amendment)[k] == v for k, v in hashes.items())
    return rows


def slots(status, maximum):
    if status['stage'] == 'complete' and status['confirmation'] == 600:
        return maximum
    if (status['stage'] == 'confirmation' and not status['failed'] and status['pending'] == 0
            and status['completed'] + len(status['running']) == 600 and time.time() - status['at'] < 30):
        return max(0, maximum - len(status['running']))
    return 0


def progress(stage, done, active, primary, errors):
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    runner.write(ROOT / 'status.json', dict(stage=stage, completed=done, total=60, running=active,
                 pending=60-done-active-len(errors), failed=errors, primary_completed=primary, updated=now))
    tmp = ROOT / '实验进度.tmp'
    tmp.write_text(f'# 12.4 ms、5% 追加压力实验\n\n{stage}。主实验已完成 {primary}/600 场。\n\n'
                   f'追加实验已完成 **{done}/60** 场，正在运行 {active} 场，失败 {len(errors)} 场。\n\n'
                   f'同一批 30 个世界，每个世界运行 I、RC 各一次。更新时间 {now}。\n')
    tmp.replace(ROOT / '实验进度.md')


def summarize(rows):
    from routeclock.evaluation import paired_bootstrap
    import pandas as pd
    records = []
    reports = []
    for cost in COSTS:
        for row in pressure(rows, cost):
            path = Path(row['out'])
            assert (path / 'completed.json').exists()
            runner.audit(row, path)
            m = read(path / 'metrics.json')
            records.append(dict(world=m['task'], method=row['method'], cost_ms=cost, net_mbps=m['net_mbps'],
                           worst10_mbps=m['worst10_mbps'], sensing_pct=100*m['sensing_fraction'],
                           hold_fail_pct=100*m['hold_failures']/m['hold_segments']))
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / 'pressure_worlds.csv', index=False)
    for metric in ('net_mbps', 'worst10_mbps', 'sensing_pct', 'hold_fail_pct'):
        table = frame.pivot(index='world', columns=['cost_ms','method'], values=metric)
        assert len(table) == 30 and not table.isna().any().any()
        for cost in COSTS:
            rc, baseline = table[cost, 'RC'], table[cost, 'I']
            reports.append(dict(cost_ms=cost, metric=metric, rc_mean=rc.mean(), i_mean=baseline.mean(),
                                **paired_bootstrap(rc, baseline)))
        if metric == 'net_mbps':
            gain = lambda c: table[c, 'RC'] - table[c, 'I']
            runner.write(ROOT / 'net_gain_change_12.4_vs_9.3.json', paired_bootstrap(gain(12.4), gain(9.3)))
    pd.DataFrame(reports).to_csv(ROOT / 'pressure_paired.csv', index=False)
    artifacts = [p for p in ROOT.rglob('*') if p.is_file() and (
                 p.parts[len(ROOT.parts)] in ('test','models') or p.suffix == '.csv' or p.name in
                 ('manifest.json','amendment.json','net_gain_change_12.4_vs_9.3.json','ch6_pressure_extension.py')) and
                 not any('.work.' in x for x in p.parts)]
    runner.write(ROOT / 'sha256.json', {str(p.relative_to(ROOT)): runner.digest(p) for p in sorted(artifacts)})


def main():
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    primary_rows = read(PRIMARY / 'manifest.json')['confirmation']
    rows = register(primary_rows)
    runner.ROOT = ROOT
    pending = deque()
    for row in rows:
        path = Path(row['out'])
        if (path / 'completed.json').exists():
            runner.audit(row, path)
        else:
            pending.append(row)
    done = len(rows) - len(pending)
    maximum = Admission().workers
    active, errors = {}, []
    primary = 0
    try:
        with ThreadPoolExecutor(max_workers=maximum) as pool:
            while pending or active:
                status = read(PRIMARY / 'status.json')
                primary = status.get('confirmation', status.get('completed', 0))
                while pending and len(active) < slots(status, maximum):
                    row = pending.popleft()
                    active[pool.submit(runner.run_one, row)] = row
                progress('运行中' if active else '等待主队列释放 CPU', done, len(active), primary, errors)
                if not active:
                    time.sleep(10)
                    continue
                completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in completed:
                    active.pop(future)
                    try:
                        future.result()
                        done += 1
                    except Exception as exc:
                        errors.append(str(exc))
        if errors:
            raise RuntimeError(str(errors))
        while read(PRIMARY / 'status.json')['stage'] != 'complete':
            progress('等待主实验汇总', done, 0, primary, [])
            time.sleep(10)
        summarize(primary_rows + rows)
        progress('已完成，已生成配对统计和文件校验清单', done, 0, 600, [])
    except Exception as exc:
        progress('需要检查', done, 0, primary, errors)
        (ROOT / 'logs/failure.txt').write_text(repr(exc))
        raise


if __name__ == '__main__':
    main()
