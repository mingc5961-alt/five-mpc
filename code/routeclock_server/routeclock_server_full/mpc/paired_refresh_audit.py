"""分解干预前的预测、获得新观测后的重估与真实交付；不改变在线控制策略。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .control import rate_table
from .experiment import dump_csv
from .paired_refresh import Replay


def audit_scene(source, scene):
    r = Replay(source, scene)
    c = r.c
    stop = c["steps"]-math.ceil(5/r.dt)+1
    output = []
    for k in range(c["steps"]):
        offset = k % c["slow_every"]
        plan, _ = r.plans[k-offset]
        future = plan[offset:]
        current = future[0]
        t = r.time(k)
        debt = np.array([float(row["debt_before_Mbit"]) for row in r.original[k]])
        if current < 0 and k in r.rotation and k < stop:
            uid = r.rotation[k]
            changed = future.copy()
            changed[0] = uid
            # 两个预测都用分叉前同一批已知状态和同一套慢层打分。只替换被刷新的预测包。
            profiles = r.profiles.copy()
            r.refresh(r.trackers.copy(), profiles, uid, k)
            table = rate_table(r.profiles, t, len(future), r.dt, r.cost)
            wait, _ = r.expected(table, debt, future)
            before, _ = r.expected(table, debt, changed)
            # 其他 15 架的预测、日后动作及感知成本完全相同。
            after_table = table.copy()
            after_table[:,uid,:] = rate_table([profiles[uid]], t, len(future), r.dt, r.cost)[:,0,:]
            after, _ = r.expected(after_table, debt, changed)
            narrow = r.radio.pairs.index((32, 32))
            output.append(dict(scene=scene, step=k, uid=uid,
                pre_state_first_Mbit=before[0]-wait[0], post_obs_first_Mbit=after[0]-wait[0],
                pre_state_window_Mbit=(before-wait).sum(), post_obs_window_Mbit=(after-wait).sum(),
                assumed_narrow_deadline_s=r.profiles[uid].deadlines[narrow],
                observed_narrow_deadline_s=profiles[uid].deadlines[narrow],
                assumed_target_fresh_Mbps=table[0,uid,1], observed_target_fresh_Mbps=after_table[0,uid,1]))
        # 验证原等待支路在全部 192 轮上的测量、传输时间、波束和物理交付逐机复现。
        next_debt, windows = r.execute(k, r.trackers, r.profiles, debt, future)
        old_action, old = r.old_trace(k, 1)[0]
        assert old_action == current
        np.testing.assert_allclose(windows, old, atol=1e-7, rtol=0)
        if k+1 < c["steps"]:
            np.testing.assert_allclose(next_debt, [float(x["debt_before_Mbit"]) for x in r.original[k+1]],
                                       atol=1e-7, rtol=0)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/demo"))
    parser.add_argument("--out", type=Path, default=Path("runs/demo_paired_refresh"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.source/"config.json").read_text(encoding="utf8"))
    rows = [r for scene in cfg["test_ids"] for r in audit_scene(args.source, scene)]
    dump_csv(args.out/"refresh_value_audit.csv", rows)
    print(f"{len(rows)} matched observations. Both original ledgers reproduced.")
