"""Compile a planning DAG. Frozen endpoint bundles expand after development."""
import argparse
import csv
import json
from collections import defaultdict
from graphlib import TopologicalSorter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def priority(row):
    key, phase = row[:2]
    if phase == 'dev_endpoint':
        prefix, scale = key.rsplit('.m', 1)
        scope, arm, seed, world = prefix.rsplit('.', 3)
        scenario = scope.removeprefix('dev.')
        if arm == 'N01' or (scenario == 'main' and arm in ('Q01','Q02','Q03','Q04')):
            return 0
        return 1 if scale == '0.75' else 2
    if phase == 'world':
        return 0 if key.startswith('w140') else 1 if key.startswith('w1') else 3
    if phase == 'heldout_bundle':
        return 1 if key.startswith(('core.main.Q','common.main.Q')) else 2
    if phase in ('panel','field','dev_panel') or key in ('field_input','archive_checks'):
        return 3
    if phase == 'features':
        return 0 if key.startswith(('features.M.','features.Acal.')) else 1
    return 0 if phase in ('prepare','replay','train','validation','select','quota_freeze','freeze_design','grid_watch','freeze_row') else 1


def build(capacity, protocol, primary_worlds):
    rows = {}
    graph = {}

    def add(key, phase, deps=(), shards=1, endpoints=0, partition=''):
        if key in rows:
            raise ValueError(f'duplicate task {key}')
        graph[key] = tuple(dict.fromkeys(deps))
        rows[key] = (key, phase, '|'.join(graph[key]), shards, endpoints, partition)
        return key

    engine = add('engine', 'prepare', partition='restore paid_full and shared simulator; resolve runtime config')
    legacy = add('grutrain_code', 'prepare', partition='restore legacy imports needed by grurun and exercise both original entry points')
    g1 = add('g1_bytes', 'prepare', partition='restore G1 checkpoint and tube; verify all four G1 artifacts')
    chrono = add('chrono_bytes', 'prepare', partition='verify Chronos seed-7 weights and tube')
    contract = add('design', 'prepare', partition='lock split namespace, perturbations, freeze and analysis rules')
    policy = add('policy_code', 'prepare', partition='implement fixed/joint D, masks and reward accounting')
    clocks = add('clocks', 'prepare', partition='build reference clock/mask and tests; integrated ledger acceptance follows')
    data_code = add('world_code', 'prepare', [engine, contract], partition='historical physical fingerprint and independent RNG streams')
    radio = add('closed_engine', 'prepare', [engine], partition='native radio and ledger; first fix and time accounting')
    grid_engine = add('grid_engine', 'prepare', [radio, clocks], partition='integrated grid/WAIT/mandatory requests and exact ledger')
    periodic = add('periodic_engine', 'prepare', [radio], partition='integrated staggered fixed clocks and 3 s reference')
    budget_engine = add('budget_engine', 'prepare', [radio], partition='parameterized native rates, first stagger, physical budget checks')
    g1gold = add('gold_G1', 'replay', [g1, legacy, radio, data_code], partition='archived world 0, scale 0.45')
    cgold = add('gold_Chronos', 'replay', [chrono, legacy, radio, data_code], partition='same world 0, scale 0.75')
    fields = add('field_input', 'prepare', partition='flight sources, timestamps, gaps, grouping and dev/held-out split')
    add('archive_checks', 'prepare', partition='independent archived receipts, reference clock and historical panel')
    bound = protocol['split_id_ranges_inclusive']
    worlds = {}
    for split in ('train', 'train_validation', 'tube_fit', 'tube_calibration', 'development', 'confirmation'):
        lo, hi = bound[split]
        if split == 'confirmation':
            hi = lo + primary_worlds - 1
        for w in range(lo, hi+1):
            worlds[w] = add(f'w{w}', 'world', [data_code], partition='one physical trajectory/noise/shadow bundle')
    stress_worlds = {}
    stress_scenarios = sorted({r['scenario'] for r in capacity if r['stage']=='stress'})
    stress_lo, stress_hi = bound['stress_family']
    for w in range(stress_lo, stress_hi+1):
        streams = add(f'streams{w}', 'world', [data_code], partition='common random streams for four paired views')
        for sc in sorted(stress_scenarios, key=lambda s:s.endswith('detour')):
            deps = [streams]
            if sc.endswith('detour'):
                deps.append(stress_worlds['structured_9_reference', w])
            stress_worlds[sc,w] = add(f'{sc}.w{w}', 'world', deps)

    def split_worlds(split):
        lo, hi = bound[split]
        if split == 'confirmation':
            hi = lo + primary_worlds - 1
        return [worlds[i] for i in range(lo, hi+1)]

    train = add('train_inputs', 'inputs', split_worlds('train'), partition='completed training split for model fitting')
    val = add('validation_inputs', 'inputs', split_worlds('train_validation'))
    models = {'Aorig':g1gold, 'F0orig':add('F0orig', 'model', [g1gold]), 'Chronos':cgold}
    center_inputs = {'Acal':[g1gold], 'M':[g1gold], 'Fstar':[g1gold,train,val],
                     'RAFT':[engine,train,val], 'IMM':[engine]}
    for name, req in center_inputs.items():
        center = add('center.'+name, 'model', req)
        features = defaultdict(list)
        for split in ('tube_fit','tube_calibration'):
            for w in split_worlds(split):
                features[split].append(add(f'features.{name}.{w}', 'features', [w, center],
                                           partition='extract per-world causal features/labels without fitting on calibration'))
        tube = add('tube.'+name, 'model', features['tube_fit'])
        models[name] = add('cal.'+name, 'model', [tube,*features['tube_calibration']],
                           partition='frozen width head and whole-path coverage calibration')
    threshold = add('imm_threshold', 'model', [models['IMM'], models['Acal'], grid_engine,'quota.B3', *split_worlds('development')],
                    partition='select risk trigger on development; auxiliary tuning cost outside endpoint total')
    clocks_ready = add('D_environment', 'prepare', [grid_engine,policy,worlds[1000],models['Acal'],models['M']],
                       partition='two predictive environments, action masks, WAIT and additive reward acceptance')

    def shared_events(sc, w, seed, r, wid, extra=()):
        refresh = r['refresh']
        if not refresh.startswith('dqn_'):
            return []
        mode = refresh[4:]
        key = f'events.{mode}.{sc}.w{w}.{seed}'
        if key not in rows:
            deps = [wid, grid_engine, f'D.{mode}.{seed}', *extra]
            if r['quota'] in ('Kstar','K(B)'):
                deps.append('quota.B'+r['budget_s'])
            add(key, 'event_stream', deps, partition='freeze events for this schedule and verify hash on independent live path')
        return [key]

    dev = defaultdict(list)
    families = set()
    endpoint_rows = []
    for r in capacity:
        if r['world_first']!='1400':
            continue
        sc, row = r['scenario'],r['row']
        b = r['budget_s']
        for seed in r['scheduler_seeds'].split('/'):
            for w in range(int(r['world_first']), int(r['world_last'])+1):
                for m in r['multiplier_rule'].split('/'):
                    deps = [worlds[w],radio,models[r['predictor']]]
                    if r['clock'] in ('fixed_count','fixed_success','grid_free'):
                        deps.append(grid_engine)
                    if r['period']:
                        deps.append(periodic)
                    if sc.startswith('B'):
                        deps.append(budget_engine)
                    if r['quota'] in ('Kstar','K(B)') or r['period']=='PM(B)':
                        deps.append(f'quota.B{b}')
                    if row=='Q06':
                        deps.append(threshold)
                    deps.extend(shared_events(sc,w,seed,r,worlds[w]))
                    task = add(f'dev.{sc}.{row}.{seed}.w{w}.m{m}', 'dev_endpoint', deps, endpoints=1)
                    dev[sc,row].append(task)
        families.add((sc,row[0]))
        endpoint_rows.append((sc,row))

    for sc, b in [('main','3'),('B4.5','4.5'),('B9','9')]:
        add('quota.B'+b, 'quota_freeze', dev[sc,'N01'], partition='N01 feasibility/extension, K and PM; invalidate dependents on change')

    seeds = protocol['quota']['dqn_seeds']
    for mode, candidates in [('fixed',3),('joint',6)]:
        req = [clocks_ready,train,'quota.B3']
        if mode=='joint':
            req += ['quota.B4.5','quota.B9']
        pilots = []
        for c in range(candidates):
            fit = add(f'pilot.{mode}.{c}', 'train', req, partition='one candidate seed 53, 16 rollouts and 4000 updates')
            pilots.append(add(f'pilot_val.{mode}.{c}', 'validation', [fit,val], 16, partition='8 worlds x 2 predictors'))
        tuning = add('hyper.'+mode, 'select', pilots, partition='one set of hyperparameters shared by all final seeds')
        for seed in seeds:
            previous = tuning
            tests = []
            for step in (10000,20000,30000):
                previous = add(f'train.{mode}.{seed}.{step}', 'train', [previous], partition='continue the same optimizer/RNG/replay state; atomic checkpoint')
                tests.append(add(f'val.{mode}.{seed}.{step}', 'validation', [previous,val], 32, partition='16 worlds x 2 predictors'))
            add(f'D.{mode}.{seed}', 'select', tests, partition='choose immutable checkpoint using validation rules')

    for sc,family in sorted(families):
        members = sorted(row for scene,row in endpoint_rows if scene == sc and row.startswith(family))
        rule = ('WATCH ' + ','.join(members) + ' in ' + sc +
                ': emit extended grid on one fully assessed infeasible row, or base grid when all rows have a feasible point. '
                'Each endpoint mean requires all dev worlds and scheduled seeds. Await evidence without holding a worker.')
        add(f'grid.{sc}.{family}', 'grid_watch', [contract], partition=rule)
    for sc,row in endpoint_rows:
        deps = [f'grid.{sc}.{row[0]}',*dev[sc,row]]
        add(f'wp.{sc}.{row}', 'freeze_row', deps, partition='await any extra endpoints for this row; seal final scales and weights, seed-averaged on development')
    size = add('sample_size', 'freeze_design', [contract,*[f'wp.main.Q0{i}' for i in range(1,5)]],
               partition='world-paired dev variance; register N=24/32/48 before accessing confirmation')
    quota_lock = add('quota_versions', 'freeze_design', [f'wp.{sc}.N01' for sc in ('main','B4.5','B9')],
                     partition='all final K/PM hashes agree with active configs, D models and dependent receipts; requeue changed descendants')
    lock = add('analysis_lock', 'freeze_design', [size,g1gold,cgold,quota_lock],
               partition='seal analysis code, world IDs and protocol; no held-out metrics to training processes')

    for r in capacity:
        if r['world_first']=='1400':
            continue
        row,sc,stage = r['row'],r['scenario'],r['stage']
        wp = f"wp.{sc if stage in ('K','B') else 'main'}.{row}"
        first, last = int(r['world_first']), int(r['world_last'])
        if sc=='main' and row in ('Q01','Q02','Q03','Q04','Q05') and stage in ('core','common'):
            last = first + primary_worlds - 1
        for w in range(first,last+1):
            for seed in r['scheduler_seeds'].split('/'):
                wid = stress_worlds[sc,w] if stage=='stress' else worlds[w]
                n = int(r['max_endpoints_per_seed_world'])
                events = shared_events(sc,w,seed,r,wid,[lock])
                add(f'{stage}.{sc}.{row}.{seed}.w{w}', 'heldout_bundle', [lock, wp, wid, *events], n, n,
                    'expand frozen scale(s), common=0.75 or stress=0.75; deduplicate canonical run keys')

    for name in ('Acal','M','Fstar','RAFT','IMM','Chronos'):
        for w in split_worlds('development'):
            add(f'dev_panel.{name}.{w}', 'dev_panel', [w,models[name]], partition='coverage, angles and acceptance on development; start by model')
    snapshots = add('snapshot_heads', 'model', [train,val], partition='age/no-age snapshot heads')
    for name, frozen in [('Acal',models['Acal']),('S','center.M'),('Fstar',models['Fstar']),
                         ('RAFT',models['RAFT']),('Chronos',cgold),('IMM',models['IMM']),('snapshots',snapshots)]:
        deps = [frozen,lock]
        row_for_hold = {'Acal':'Q01', 'S':'Q02','Fstar':'Q05','RAFT':'Q07','Chronos':'Q08','IMM':'Q10'}
        if name in row_for_hold:
            deps.append('wp.main.'+row_for_hold[name])
        for w in split_worlds('confirmation'):
            add(f'panel.{name}.{w}', 'panel', [*deps,w], partition='share paid prefix; age/horizon and natural service subset together')
    field_lock = add('field_lock','freeze_design',[fields,contract],partition='freeze flight split, geometry, gap and metrics before scoring')
    for name in ('Acal','M','Fstar','RAFT'):
        add('field_geometry.'+name,'field',[field_lock,'center.'+name],
            partition='shard by full flight after QC; keep held-out scores sealed until model and shared analysis decisions are locked')

    missing = {d for deps in graph.values() for d in deps if d not in graph}
    if missing:
        raise ValueError(f'unresolved prerequisites: {sorted(missing)}')
    ordered = [rows[k] for k in TopologicalSorter(graph).static_order()]
    expected = sum(int(r['max_endpoint_runs']) for r in capacity)
    expected += (primary_worlds - 24) * 27
    if sum(r[4] for r in ordered) != expected:
        raise ValueError('endpoint counts differ from the capacity sheet')
    return ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, help='new CSV path; otherwise print the counts')
    p.add_argument('--primary-worlds', type=int, choices=(24,32,48), default=24,
                   help='planning scenario; use the registered development decision before dispatching')
    args = p.parse_args()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    with (ROOT/'SERVER_CAPACITY.csv').open(encoding='utf-8-sig', newline='') as f:
        tasks = build(list(csv.DictReader(f)), protocol, args.primary_worlds)
    if args.out:
        with args.out.open('x',encoding='utf-8-sig',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id','phase','dependencies','independent_shards','max_endpoint_runs','partition_or_release_rule','priority'])
            writer.writerows((*task,priority(task)) for task in tasks)
    for phase in dict.fromkeys(r[1] for r in tasks):
        group = [r for r in tasks if r[1]==phase]
        print(f'{phase:16} {len(group):4} nodes {sum(r[3] for r in group):4} shards {sum(r[4] for r in group):4} endpoint slots')


if __name__=='__main__':
    main()
