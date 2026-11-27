"""Read compatible per-origin measurements from this frozen experiment."""
from base import BASE, tag, sha
import json
import numpy as np

CPU_MODELS = {'IMM', 'T8', 'T180', 'G8', 'G180', 'T8P', 'G8P', 'Joint07'}
PANEL_SOURCE = {'C': 'G8P', 'T8': 'T8P', 'G8': 'G8P', 'Cexternal': 'Cnative'}


def read_panel(args):
    key = 'panel__' + tag(args)
    receipt = BASE / 'receipts' / (key + '.json')
    archive = receipt.with_suffix('.npz')
    if not receipt.exists() or not archive.exists():
        return None
    rec = json.loads(receipt.read_bytes())
    if rec['id'] != key or rec['args'] != args or rec['data']['n'] != 960:
        return None
    with np.load(archive) as f:
        data = {name: f[name] for name in f.files}
    if data['errors'].shape != (960, len(data['names']), 4):
        return None
    if list(data['names']) != rec['data']['names'] or data['keys'].shape != (960, 5):
        return None
    if not np.array_equal(data['keys'][:, 0].astype(int), np.tile(np.arange(16), 60)):
        return None
    return data, {'task': key, 'npz_sha256': sha(archive)}


def same_aircraft(left, right):
    if (left.cfg != right.cfg or left.dt != right.dt or left.n != right.n
            or left.namespace != right.namespace or left.mission != right.mission
            or not np.array_equal(left.times, right.times)):
        return []
    fields = ('positions', 'velocities', 'noise', 'groups', 'references', 'progress', 'detour_windows')
    return [j for j in range(left.n) if all(np.array_equal(
        getattr(left, field)[j], getattr(right, field)[j]) for field in fields)]


def paired_rows(args, target_world, load_world):
    if args['a'] not in CPU_MODELS or args['w'] < 3000:
        return None
    old_args = dict(args, w=args['w'] - 3000)
    src = read_panel(old_args)
    if src is None:
        return None
    shared = same_aircraft(load_world(old_args['w']), target_world)
    if not shared:
        return None
    data, provenance = src
    provenance = dict(provenance, aircraft=shared, origins=60 * len(shared))
    return data, shared, provenance


def sibling_rows(args):
    model = args['a']
    if args['mode'] != 'all' or model not in PANEL_SOURCE:
        return None
    src = read_panel(dict(args, a=PANEL_SOURCE[model]))
    if src is None:
        return None
    data, provenance = src
    if model == 'C':
        route = np.flatnonzero(data['names'] == 'G8P-route')
        if len(route) != 1:
            return None
        i = int(route[0])
        return data['keys'], data['errors'][:, i:i+1], dict(provenance, field='G8P-route', origins=960)
    field = 'errors' if model == 'Cexternal' else 'standalone_errors'
    if field not in data or data[field].shape != (960, 1, 4):
        return None
    return data['keys'], data[field], dict(provenance, field=field, origins=960)
