"""Recheck the published diagnostic's four synthetic worlds with only the MCS margin varied."""
import json
import sys
from unittest.mock import patch
import numpy as np
from pathlib import Path

ROOT = Path('/data/chen/routeclock_runs/confirm_0db')
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'work/code/routeclock_refine')]
from advanced_tubes import Predictor, get_bundle
from refine import network, world_cache
from run_baselines import NAMESPACE, paired_world
from routeclock.config import load_config
from routeclock.radio import Radio
from runtime import remaining_runs as runtime


def replay(world_id):
    runtime.OUT = world_cache.BASE = ROOT / 'work/baseline_cache'
    network.paired_world = paired_world
    runner = runtime.Runner()
    dataset = ROOT / 'results'
    cfg = load_config()
    cfg['simulation']['sensing_budget'] = runtime.BUDGET['T']
    world = paired_world(cfg, NAMESPACE, world_id)
    net = Predictor('Chronos-2', 7)
    params = get_bundle(net, 7)

    for name in ('RC', 'Chronos-2'):
        if name == 'RC':
            archive = dataset / 'baselines' / f'{world_id:03}_T_RC/complete.json'
        else:
            from parallel_frontier import frontier
            assert (ROOT / 'state/frontier_verified.json').exists()
            network.frontier = frontier
            archive = dataset / 'confirmation/Chronos-2/7' / f'{world_id:03}_T_native/complete.json'
        record = json.loads(archive.read_text())
        ref = record['result']
        centers = []
        original_run = network.run_world
        def capture(config, physical, model, bundle, tube):
            initial = model.batch
            def first(histories):
                model.batch = initial
                estimates = initial(histories)
                centers.append(np.stack([c for c, _ in estimates]))
                return estimates
            model.batch = first
            return original_run(config, physical, model, bundle, tube)

        for margin in (0, 5):
            if name == 'RC':
                row = dict(record['frozen_config'], mcs_margin_db=margin)
                for key in ('namespace', 'inputs'):
                    del row[key]
                with patch.object(network, "run_world", capture):
                    result = runner.execute(row)
            else:
                assert record['center_sha256'] == net.sha and record['tube'] == 'native'
                radio = Radio(cfg)
                radio.threshold += margin - 5
                with runtime.controller(radio, 10):
                    result, _ = capture(cfg, world, net, params, 'native')
            assert result['probes_per_user'] == 9000
            assert abs(sum(result['ledger_s']) - 180) < 1e-6
            if margin == 5:
                np.testing.assert_allclose(centers[0], centers[1], atol=1e-8, rtol=0)
            row = dict(world=world_id, model=name, archive_G_mbps=ref['G_mbps'],
                       initial_center_sha256=result['initial_center_sha256'], archive_center_sha256=ref['initial_center_sha256'], mcs_margin_db=margin, G_mbps=result['G_mbps'],
                       risk_pct=100*result['R'], updates=result['updates'], archive_0db=str(archive))
            (OUT / f'{world_id:03}_{name}_{margin}.json').write_text(json.dumps(row, indent=2))
            print(row, flush=True)


if __name__ == '__main__':
    replay(int(sys.argv[1]))
