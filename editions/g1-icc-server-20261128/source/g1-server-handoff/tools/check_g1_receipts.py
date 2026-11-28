"""Check published G1 receipts and compute a development-only periodic reference."""

import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean, median_low


USERS = 16
DURATION_US = 180_000_000
COST_US = 3_100
FIRST_SLOT_SPACING_US = 186_000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def counts_at_period(period_ms):
    period_us = period_ms * 1_000
    return [
        1 + (DURATION_US - COST_US - uid * FIRST_SLOT_SPACING_US) // period_us
        for uid in range(USERS)
    ]


def check_receipt(receipt):
    result = receipt["result"]
    events = receipt["events"]
    expected_first = [
        (uid, (uid * FIRST_SLOT_SPACING_US + COST_US) / 1e6)
        for uid in range(USERS)
    ]
    assert len(events) >= USERS
    for (uid, end), (expected_uid, expected_end) in zip(events[:USERS], expected_first):
        assert uid == expected_uid and isclose(end, expected_end, abs_tol=1e-8)
    assert result["probes_per_user"] == 9000 and len(result["bits"]) == USERS
    assert isclose(sum(result["ledger_s"]), 180.0, rel_tol=0, abs_tol=1e-6)
    assert isclose(result["ledger_s"][0], len(events) * COST_US / 1e6, abs_tol=1e-7)
    assert result["ledger_s"][0] <= 3.00001
    assert isclose(sum(result["bits"]) / 180e6, result["G_mbps"], abs_tol=1e-6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="path to the five-mpc repository root")
    args = parser.parse_args()
    work = args.repo / "editions/g1-paid-cold-start-20261127/evidence/work"
    run = work / "factorial_run"
    weights = read_json(run / "freeze.json")
    original = read_json(run / "run_confirm_0_G1_0.45.json")
    anchor = check_receipt(original)
    print(f"Anchor: {len(original['events'])} events, {anchor['G_mbps']:.5f} Mbps, "
          f"risk {100 * anchor['R']:.5f}%, sensing {anchor['ledger_s'][0]:.4f} s, "
          f"usable narrow beam {anchor['narrow_good_data_s']:.4f} s")

    dev_costs = []
    differences = []
    dev_worlds = set()
    confirm_worlds = set()
    for k in range(8):
        cost = 0.0
        rates = {}
        dev_ids, confirm_ids = set(), set()
        for arm in ("G1", "C"):
            rates[arm] = 0.0
            assert isclose(sum(weights[arm].values()), 1.0)
            for scale, probability in weights[arm].items():
                suffix = f"{k}_{arm}_{float(scale):.2f}.json"
                receipt = read_json(run / f"run_confirm_{suffix}")
                confirm_ids.add(receipt["world"])
                rates[arm] += probability * check_receipt(receipt)["G_mbps"]
                if arm == "G1":
                    dev = read_json(work / "g1mix7" / f"run_test_{suffix}")
                    dev_ids.add(dev["world"])
                    cost += probability * check_receipt(dev)["ledger_s"][0]
        assert len(dev_ids) == len(confirm_ids) == 1
        dev_worlds.update(dev_ids)
        confirm_worlds.update(confirm_ids)
        dev_costs.append(cost)
        differences.append(rates["G1"] - rates["C"])
    assert len(dev_worlds) == len(confirm_worlds) == 8 and dev_worlds.isdisjoint(confirm_worlds)
    print(f"Frozen G1 minus Chronos: {mean(differences):+.5f} Mbps, "
          f"{sum(d > 0 for d in differences)}/8 positive worlds")
    target_events = mean(dev_costs) * 1e6 / COST_US
    periods = range(3000, 6001)
    errors = {p: abs(sum(counts_at_period(p)) - target_events) for p in periods}
    lowest = min(errors.values())
    matched = median_low([p for p in periods if errors[p] == lowest])
    print(f"Frozen G1 development sensing target: {mean(dev_costs):.8f} s")
    for period in (3000, matched):
        counts = counts_at_period(period)
        total = sum(counts)
        print(f"Period {period / 1000:.3f} s: {total} events, {total * COST_US / 1e6:.4f} s, "
              f"{min(counts)}-{max(counts)} events per user")


if __name__ == "__main__":
    main()
