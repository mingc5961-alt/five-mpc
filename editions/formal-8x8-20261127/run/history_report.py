"""Paired 8/8 history-window comparisons at the fixed tight budget and 1.00 scale."""
import json

import numpy as np
import pandas as pd

from base import BASE

ANGLES = ['angle_1s', 'angle_2s', 'angle_5s', 'angle_10s']
CLOSED = ['closed_1s', 'closed_2s', 'closed_5s', 'closed_10s']
RADIO = ['risk_pct', 'throughput_mbps', 'sensing_s']
METHODS = ['C', 'Cnative', 'G8P', 'IMM']
WORLDS = [0, 100, 101, 102, 103, 104]


def paired_delta(data, baseline, keys, metrics, suffix):
    ref = baseline[keys + metrics].rename(columns={m: m + suffix for m in metrics})
    result = data.merge(ref, on=keys, how='left', validate='many_to_one', indicator=True)
    if not result['_merge'].eq('both').all():
        raise ValueError(f'Missing reference for {suffix}')
    for m in metrics:
        result[m + suffix] = result[m] - result[m + suffix]
    return result.drop(columns='_merge')


def group_rows(archive, row, names, world_keys):
    with np.load(archive) as data:
        keys, errors = data['keys'], data['errors']
        if list(data['names']) != names or errors.shape != (960, len(names), 4):
            raise ValueError(f'Invalid panel {archive}')
    previous = world_keys.setdefault(row['world'], keys)
    if not np.array_equal(keys, previous):
        raise ValueError(f'Unpaired origins in {archive}')
    groups = {'all': np.ones(len(keys), dtype=bool), 'random': keys[:, 2] == 'random',
              'revisit': keys[:, 2] == 'revisit', 'detour': keys[:, 2] == 'detour',
              'early_detour': keys[:, 3] == 'True'}
    for group, mask in groups.items():
        for index, name in enumerate(names):
            values = errors[mask, index].mean(axis=0) if mask.any() else [np.nan]*4
            yield dict(row, variant=name, group=group, origins=int(mask.sum()), **dict(zip(ANGLES, values)))


def history_check(task, dest):
    cells, panels, angle_rows, world_keys = [], [], [], {}
    for key in task['deps']:
        path = BASE/'receipts'/(key+'.json')
        record = json.loads(path.read_bytes())
        a, d = record['args'], record['data']
        row = dict(model=a['a'], world=a['w'], history_s=a['H'], mode=a['mode'])
        if 'result' in d:
            r = d['result']
            assert a['m'] == 1 and a['b'] == 'T'
            cells.append(dict(row, transfer=a['transfer'], risk_pct=r['R']*100,
                              throughput_mbps=r['G_mbps'], sensing_s=r['ledger_s'][0],
                              **dict(zip(CLOSED, d['closed_loop_deg']))))
        else:
            panels.append(dict(row, **dict(zip(ANGLES,d['angles'][d['names'].index(a['a'])]))))
            angle_rows.extend(group_rows(path.with_suffix('.npz'), row, d['names'], world_keys))

    keys = ['model', 'world', 'history_s', 'mode']
    cells = pd.DataFrame(cells)
    panels = pd.DataFrame(panels)
    expected = {(a,w,h,'all') for a in METHODS for w in WORLDS for h in (30,120,180)}
    standard = cells[~cells.transfer].drop(columns='transfer')
    fixed = cells[cells.transfer].drop(columns='transfer')
    def check_grid(data, required):
        grid = list(data[keys].itertuples(index=False,name=None))
        if len(grid) != len(required) or set(grid) != required:
            raise ValueError('Incomplete or duplicate history comparison')
    check_grid(standard, expected)
    check_grid(fixed, {(a,w,30,'all') for a in ('G8P','Cnative') for w in WORLDS})
    check_grid(panels, expected | {(a,w,60,'all') for a in METHODS for w in WORLDS}
               | {('G8P',w,30,'route') for w in WORLDS})

    paired = standard.merge(panels, on=keys, validate='one_to_one')
    paired = paired_delta(paired, paired[paired.history_s==180], ['model','world'],
                          RADIO+CLOSED+ANGLES, '_minus_180')
    paired.to_csv(dest/'worlds.csv', index=False)
    conf = paired[paired.world>=100].drop(columns=['world','mode'])
    conf.groupby(['model','history_s']).mean().to_csv(dest/'confirmation_means.csv')

    tubes = paired_delta(fixed, standard, keys, RADIO+CLOSED, '_minus_recalibrated')
    tubes = paired_delta(tubes, standard[standard.history_s==180], ['model','world'],
                         RADIO+CLOSED, '_minus_180')
    tubes.to_csv(dest/'tube_worlds.csv', index=False)
    tubes[tubes.world>=100].drop(columns=['world','mode','history_s']).groupby('model').mean().to_csv(
        dest/'tube_confirmation.csv')

    angles = pd.DataFrame(angle_rows)
    reference = angles[(angles.history_s==180) & (angles['mode']=='all')]
    angles = paired_delta(angles, reference, ['world','model','variant','group'], ANGLES, '_minus_180')
    chronos = angles[(angles.variant=='Cnative') & (angles['mode']=='all')]
    angles = paired_delta(angles, chronos, ['world','history_s','group'], ANGLES, '_minus_chronos')
    angles.to_csv(dest/'angles_worlds.csv', index=False)
    grouped = angles[angles.world>=100].groupby(['model','variant','history_s','mode','group'])
    metrics = ANGLES+[m+s for s in ('_minus_180','_minus_chronos') for m in ANGLES]
    means = grouped[metrics].mean()
    means['origins'] = grouped.origins.sum()
    means['worlds_with_origins'] = grouped.origins.agg(lambda n: (n>0).sum())
    means.to_csv(dest/'angles_confirmation.csv')
    route = angles[(angles.model=='G8P') & (angles.history_s==30)]
    route_diff = paired_delta(route[route['mode']=='route'], route[route['mode']=='all'],
                              ['world','variant','group'], ANGLES, '_minus_all_short')
    route_diff.to_csv(dest/'route_only_worlds.csv', index=False)
    return dict(world_rows=len(paired), confirmation_worlds=5, tube_pairs=len(tubes), folder=str(dest))
