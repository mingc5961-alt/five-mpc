"""在旧 demo 的真实 MPC 状态上做单次动作干预。运行 python -m mpc.paired_refresh。"""
import argparse
import copy
import csv
import json
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .control import fast_plan, one_step_allocation, rate_table, sensing_clock, slow_plan
from .data import publish
from .experiment import dump_csv, settle
from .forecast_training import load_models
from routeclock.radio import Radio


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Replay:
    def __init__(self, root, scene):
        self.c = json.loads((root / "config.json").read_text(encoding="utf8"))
        self.model = load_models(root)
        with (root / f"test_{scene}.pkl").open("rb") as stream:
            self.world, self.trackers, packages = pickle.load(stream)
        self.profiles = [self.model.profile(p) for p in packages]
        self.radio = Radio(self.world.cfg)
        sim = self.world.cfg["simulation"]
        self.cost, self.switch = sim["observation_cost_s"], sim["beam_switch_s"]
        self.dt, self.n = self.c["dt"], self.world.n
        self.clock = sensing_clock(self.c["steps"], self.dt, self.c["sensing_budget"], self.cost)
        folder = root / f"scene_{scene}"
        self.plans = {int(r["step"]): ([int(a) for a in r["actions"].split()], float(r["objective"]))
                      for r in read_csv(folder / "joint_mpc/plans.csv")}
        ledger = read_csv(folder / "joint_mpc/ledger.csv")
        self.original = [ledger[k:k+self.n] for k in range(0, len(ledger), self.n)]
        self.rotation = {int(r["step"]): int(r["uid"]) for r in read_csv(folder / "rotation/ledger.csv")
                         if int(r["observed"])}

    def time(self, step):
        return self.c["warmup_s"] + step*self.dt

    def refresh(self, trs, profiles, uid, step):
        # 独立复制发生变化的跟踪器。其余无人机的既往观测与预测以只读方式共享。
        tr = copy.deepcopy(trs[uid])
        tr.observe(self.world.observe(uid, self.time(step)+self.cost))
        trs[uid] = tr
        profiles[uid] = self.model.profile(publish(tr, self.model.gate))

    def execute(self, k, trs, profiles, debt, future):
        """与正式闭环共用快层、观测器和物理结算；保留传输窗口供精确截取。"""
        action = future[0]
        t = self.time(k)
        if action >= 0:
            assert self.clock[k]
            self.refresh(trs, profiles, action, k)
        h = min(self.c["steps"]-k, self.c["fast_horizon"])
        scheduled = (future + [-1]*h)[:h]
        airtime, beams = fast_plan(profiles, t, debt, scheduled, self.c, self.cost, self.switch)
        cursor = t + (self.cost if action >= 0 else 0.)
        received = np.zeros(self.n)
        windows = []
        for uid, (seconds, beam) in enumerate(zip(airtime, beams)):
            cursor += self.switch
            pr = profiles[uid]
            pair = self.radio.pairs[beam]
            bits, expired, _ = settle(self.world, self.radio, uid, pr, pair, cursor, seconds, pr.deadlines[beam])
            received[uid] = bits
            windows.append((cursor, seconds-expired, bits, pair == (32, 32)))
            cursor += seconds
        assert abs(cursor-t-self.dt) < 1e-8
        debt = np.maximum(0., debt+self.c["target_network_mbps"]*self.dt/self.n-received)
        return debt, windows

    def rollout(self, start, initial, parent_plan, target, locked=False):
        trs, profiles, debt = initial
        trs, profiles, debt = trs.copy(), profiles.copy(), debt.copy()
        every = self.c["slow_every"]
        offset = start % every
        plan = parent_plan[offset:].copy()
        plan[0] = target
        origin = start
        steps = len(plan) if locked else math.ceil(5/self.dt)
        trace = []
        for k in range(start, start+steps):
            if not locked and k > start and k % every == 0:
                horizon = min(self.c["steps"]-k, self.c["slow_horizon"])
                plan, _ = slow_plan(profiles, self.time(k), debt, horizon, self.c, self.cost,
                                    self.switch, self.clock[k:k+horizon])
                origin = k
            future = plan[k-origin:]
            debt, windows = self.execute(k, trs, profiles, debt, future)
            trace.append((future[0], windows))
        return trace

    def old_trace(self, start, steps):
        trace = []
        for k in range(start, start+steps):
            rows = self.original[k]
            action = next((int(r["uid"]) for r in rows if int(r["observed"])), -1)
            cursor = self.time(k)+(self.cost if action >= 0 else 0.)
            windows = []
            for uid, r in enumerate(rows):
                assert int(r["uid"]) == uid
                cursor += self.switch
                seconds = float(r["data_s"])
                windows.append((cursor, seconds-float(r["expired_s"]), float(r["delivered_Mbit"]),
                                int(r["beam_y"]) == int(r["beam_z"]) == 32))
                cursor += seconds
            trace.append((action, windows))
        return trace

    def expected(self, table, debt, plan):
        """按原慢层评分原样重算，比特与欠账分开保存，不能把评分差当成比特差。"""
        last = np.zeros(self.n, int)
        bits, scores = [], []
        for j, a in enumerate(plan):
            if a >= 0:
                last[a] = j+1
            rates = table[j, np.arange(self.n), last][None, :]
            available = np.array([self.dt-self.n*self.switch-self.cost*(a >= 0)])
            due = debt+self.c["target_network_mbps"]*self.dt/self.n
            b, new = one_step_allocation(rates, available, due[None, :],
                                         self.c["min_airtime_fraction"], self.c["deficit_weight"])
            bits.append(b[0])
            scores.append(b[0]+self.c["deficit_weight"]*(debt.sum()-new.sum()))
            debt = new[0]
        return np.array(bits), np.array(scores)


def totals(trace, t0, horizon, dt):
    # 每段沿用原链路账本的中点速率，仅计截止时刻之前的有效发送时间。
    end = t0+horizon
    bits = narrow = 0.
    observations = 0
    for j, (action, windows) in enumerate(trace):
        if t0+j*dt >= end-1e-10:
            break
        observations += int(action >= 0)
        for start, valid, delivered, is_narrow in windows:
            used = min(valid, max(0., end-start))
            if used > 0:
                bits += delivered*(used/valid)
                narrow += used*is_narrow
    return np.array([bits, narrow*1000, observations])


def run_chunk(job):
    root, out, scene, low, high = job
    r = Replay(root, scene)
    c = r.c
    debt = np.zeros(r.n)
    rows = []
    for k in range(high):
        entry = r.original[k]
        np.testing.assert_allclose(debt, [float(i["debt_before_Mbit"]) for i in entry], atol=1e-7, rtol=0)
        offset = k % c["slow_every"]
        if offset == 0:
            plan, score = r.plans[k]
            table = rate_table(r.profiles, r.time(k), len(plan), r.dt, r.cost)
            parent_debt = debt.copy()
            predicted, utility = r.expected(table, parent_debt, plan)
            np.testing.assert_allclose(utility.sum(), score, atol=1e-6, rtol=0)
        if low <= k and plan[offset] == -1 and k in r.rotation:
            uid = r.rotation[k]
            changed = plan.copy()
            changed[offset] = uid
            refresh_pred, refresh_util = r.expected(table, parent_debt, changed)
            predicted_delta = (refresh_pred-predicted)[offset:]
            remain = len(plan)-offset
            matched = remain*r.dt
            initial = (r.trackers, r.profiles, debt)
            # 等待+同一 MPC 已逐时隙留在原始账本；刷新支路只置换第一项控制输入。
            wait = r.old_trace(k, math.ceil(5/r.dt))
            fresh = r.rollout(k, initial, plan, uid)
            # 额外检验：冻结发布时其余慢层动作，消除重规划差异，快层仍照常求解。
            lock_wait = r.rollout(k, initial, plan, -1, locked=True)
            lock_fresh = r.rollout(k, initial, plan, uid, locked=True)
            row = dict(scene=scene, step=k, uid=uid, offset=offset, time_since_service_s=k*r.dt,
                       matched_s=matched, pred_first_Mbit=predicted_delta[0],
                       pred_window_Mbit=predicted_delta.sum(),
                       pred_window_utility=(refresh_util-utility).sum())
            for name, h, a, b in (("first", r.dt, wait, fresh), ("matched", matched, wait, fresh),
                                  ("locked", matched, lock_wait, lock_fresh),
                                  ("1s", 1., wait, fresh), ("3s", 3., wait, fresh), ("5s", 5., wait, fresh)):
                av, bv = [totals(tr, r.time(k), h, r.dt) for tr in (a, b)]
                for metric, v, w in zip(("Mbit", "narrow_ms", "observations"), av, bv):
                    row[f"wait_{name}_{metric}"] = v
                    row[f"refresh_{name}_{metric}"] = w
                    row[f"delta_{name}_{metric}"] = w-v
            rows.append(row)
            print(f"scene={scene} step={k} paired", flush=True)
        # 对每个被用于分叉的历史状态，按旧计划重放真实单轮并与归档的每机交付核对。
        debt, windows = r.execute(k, r.trackers, r.profiles, debt, plan[offset:])
        np.testing.assert_allclose([w[2] for w in windows], [float(i["delivered_Mbit"]) for i in entry],
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose([w[1] for w in windows],
                                   [float(i["data_s"])-float(i["expired_s"]) for i in entry],
                                   rtol=0, atol=1e-9)
        assert [w[3] for w in windows] == [int(i["beam_y"]) == int(i["beam_z"]) == 32 for i in entry]
    if rows:
        dump_csv(out/f"part_{scene}_{low}.csv", rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/demo"))
    parser.add_argument("--out", type=Path, default=Path("runs/demo_paired_refresh"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    c = json.loads((args.source/"config.json").read_text(encoding="utf8"))
    args.out.mkdir(parents=True, exist_ok=True)
    stop = c["steps"]-math.ceil(5/c["dt"])+1
    jobs = [(args.source, args.out, scene, k, min(k+20, stop))
            for scene in c["test_ids"] for k in range(0, stop, 20)]
    # 四个独立仿真进程各使用单线程数值库，避免 BLAS/OpenMP 重复并行。
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_chunk, job): job for job in jobs}
        sizes = {pending[f]: f.result() for f in as_completed(pending)}
        count = sum(sizes.values())
    rows = [r for job in jobs if sizes[job] for r in read_csv(args.out/f"part_{job[2]}_{job[3]}.csv")]
    assert len(rows) == count == len({(r["scene"], r["step"]) for r in rows})
    dump_csv(args.out/"paired_events.csv", rows)
    print(f"Completed {count} events.", flush=True)


if __name__ == "__main__":
    main()
