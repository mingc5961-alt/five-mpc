"""Project the three weighted IMM mode means on the published 0.5 s grid."""
from functools import lru_cache
import numpy as np
import server_paths
from routeclock.filters import position_transitions


@lru_cache(maxsize=1)
def mean_projection():
    trans, dynamics = position_transitions(0.5)
    step = np.block([[trans[i,j]*dynamics[j] for i in range(3)] for j in range(3)])
    read = np.tile(np.eye(6)[:3], (1,3))
    rows = [read]
    for _ in range(20):
        rows.append(rows[-1] @ step)
    return np.stack(rows)
