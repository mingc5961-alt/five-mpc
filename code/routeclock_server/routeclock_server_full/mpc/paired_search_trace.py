"""只读重演原慢层的候选剪枝，检查“刷新轮询目标”是否曾被规划到窗口末端。"""
import argparse
import math
from pathlib import Path

import numpy as np
from .control import one_step_allocation, rate_table
from .experiment import dump_csv
from .paired_refresh import Replay


def trace(r, k, debt, plan, expected_score):
    c, n = r.c, r.n
    horizon = len(plan)
    table = rate_table(r.profiles, r.time(k), horizon, r.dt, r.cost)
    paths = np.empty((1, 0), int)
    last = np.zeros((1, n), int)
    debts, scores = debt[None, :], np.zeros(1)
    survivors = []
    for j in range(horizon):
        options = np.arange(-1,n) if r.clock[k+j] else np.array([-1])
        parent = np.repeat(np.arange(len(scores)), len(options))
        actions = np.tile(options, len(scores))
        active = actions >= 0
        updated = last[parent].copy()
        updated[np.flatnonzero(active), actions[active]] = j+1
        rates = table[j, np.arange(n)[None,:], updated]
        available = r.dt-r.cost*active-n*r.switch
        bits, new = one_step_allocation(rates, available, debts[parent]+c["target_network_mbps"]*r.dt/n,
                                        c["min_airtime_fraction"], c["deficit_weight"])
        value = scores[parent]+bits+c["deficit_weight"]*(debts[parent].sum(axis=1)-new.sum(axis=1))
        keep = np.argsort(-value, kind="stable")[:c["search_width"]]
        paths = np.column_stack([paths[parent[keep]], actions[keep]])
        debts, scores, last = new[keep], value[keep], updated[keep]
        survivors.append(paths)
    assert paths[0].tolist() == plan
    np.testing.assert_allclose(scores[0], expected_score, atol=1e-6, rtol=0)
    return survivors


def scene_trace(root, scene):
    r = Replay(root, scene)
    stop = r.c["steps"]-math.ceil(5/r.dt)+1
    rows = []
    for k in range(stop):
        offset = k % r.c["slow_every"]
        plan, score = r.plans[k-offset]
        if offset == 0:
            debt = np.array([float(row["debt_before_Mbit"]) for row in r.original[k]])
            survivors = trace(r, k, debt, plan, score)
        if plan[offset] < 0 and k in r.rotation:
            target = r.rotation[k]
            prefix = plan[:offset]+[target]
            # 此前动作须与最终执行计划完全相同，只对本次刷新动作作替换。
            kept = sum(bool(np.any(np.all(paths[:, :offset+1] == prefix, axis=1)))
                       for paths in survivors[offset:])
            rows.append(dict(scene=scene, step=k, uid=target, stages_survived=kept,
                        remaining_stages=len(plan)-offset, pruned_immediately=int(kept==0),
                        reached_final=int(kept==len(plan)-offset)))
        if plan[offset] >= 0:
            r.refresh(r.trackers, r.profiles, plan[offset], k)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/demo"))
    parser.add_argument("--out", type=Path, default=Path("runs/demo_paired_refresh"))
    args = parser.parse_args()
    dump_csv(args.out/"search_trace.csv", [r for s in (0,1) for r in scene_trace(args.source, s)])
