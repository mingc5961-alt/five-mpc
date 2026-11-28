"""Post-hoc grid sensitivity on the same held-out worlds; never changes primary receipts."""
import factorial_closed as fc
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

def extra(path_split, k, a, m):
    return fc.ROOT/f"sensitivity_grid_{k}_{a}_{m:.2f}.json"

def run(job):
    with patch.object(fc,"receipt",extra):
        fc.simulate(job)

if __name__=="__main__":
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(run,[("confirm",k,"G1",m) for k in range(8) for m in (.4,.85)]))
