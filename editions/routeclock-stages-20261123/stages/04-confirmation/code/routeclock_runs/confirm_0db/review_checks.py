from pathlib import Path
import json
import numpy as np
from train_vector_long import features
from train_chronos_long import context

root = Path(__file__).resolve().parent
t = np.array([0., 20., 100., 180., 200., 201.])
p = np.arange(18).reshape(-1,3).astype(float)
for build in (features, context):
    baseline = build(t,p,200.)
    perturb = p.copy()
    perturb[-1] += 1e5
    np.testing.assert_equal(build(t,perturb,200.),baseline)
    perturb = p.copy()
    perturb[0] -= 1e5
    np.testing.assert_equal(build(t,perturb,200.),baseline)
print("Future and out-of-window perturbation tests passed for VECTOR and Chronos-2")
