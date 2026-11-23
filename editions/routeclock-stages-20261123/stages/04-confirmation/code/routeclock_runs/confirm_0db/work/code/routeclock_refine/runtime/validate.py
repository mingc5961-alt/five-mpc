"""Re-run the archived validation cells with vectorized final accounting."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from refine import network
from refine.data import BASE
from runtime.fast_ledger import DeferredLedger

def validate(model):
    target = BASE / 'runtime' / 'validation_replay'
    network.Ledger = DeferredLedger
    tubes = ['native', 'lightgbm']
    network.execute(model,'validation',[0],tubes,target)
    checks = []
    for tube in tubes:
        rel = Path('validation') / model / '000' / (tube + '.json')
        original = json.loads((BASE / 'tube_network' / rel).read_text())
        replay = json.loads((target / rel).read_text())
        differences = {}
        assert original.keys() == replay.keys()
        for key in original:
            if key in ('bits','G_mbps','ledger_s','narrow_good_data_s'):
                delta = np.abs(np.asarray(original[key]) - replay[key])
                differences[key] = float(delta.max())
                np.testing.assert_allclose(original[key],replay[key],rtol=1e-11,atol=1e-8)
            elif key != 'wall_s':
                assert original[key] == replay[key], (model,tube,key)
        checks.append(dict(tube=tube,max_abs_differences=differences))
    code = Path(__file__).with_name('fast_ledger.py').read_bytes()
    path = target / (model + '_verified.json')
    path.write_text(json.dumps(dict(checks=checks,sha256=hashlib.sha256(code).hexdigest()),indent=2))

if __name__ == '__main__':
    validate(sys.argv[1])
