"""Recompute the two-world confirmation operating points using only Python's standard library."""
import itertools
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
CONFIGS = {"C0": (.35, .55, .75, 1.), "G0": (.55, .75, .90, 1.),
           "G1": (.55, .75, .90, 1.), "I1": (.55, .75, .90, 1.)}


def load_curves():
    curves = {}
    fingerprints = {}
    for arm, scales in CONFIGS.items():
        points = {}
        for scale in scales:
            rows = []
            for seed in (0, 1):
                name = f"confirm_irregular_{seed}_{arm[0]}_{arm[1]}_{scale:.3f}.json"
                record = json.loads((ROOT / "reference" / name).read_text(encoding="utf-8"))
                assert (record["case"], record["split"], record["seed"], record["m"],
                        record["arm"], record["profile"]) == ("irregular", "confirm", seed, scale, arm[0], int(arm[1]))
                assert record["fp"] == fingerprints.setdefault(seed, record["fp"])
                result = record["result"]
                assert result["probes_per_user"] == 9000 and len(result["bits"]) == 16
                assert math.isclose(sum(result["bits"]) / 180e6, result["G_mbps"])
                rows.append((100 * result["R"], result["G_mbps"], result["narrow_good_data_s"]))
            points[scale] = tuple(map(mean, zip(*rows)))
        curves[arm] = points
    return curves


def best_at(points, target):
    options = [(values, {scale: 1.0}) for scale, values in points.items()
               if math.isclose(values[0], target, abs_tol=1e-12)]
    for (a, x), (b, y) in itertools.combinations(points.items(), 2):
        if min(x[0], y[0]) < target < max(x[0], y[0]):
            p = (target - y[0]) / (x[0] - y[0])
            mixed = tuple(p * i + (1 - p) * j for i, j in zip(x, y))
            options.append((mixed, {a: p, b: 1 - p}))
    return max(options, key=lambda pair: pair[0][1])


if __name__ == "__main__":
    curves = load_curves()
    lo = max(min(row[0] for row in points.values()) for points in curves.values())
    hi = min(max(row[0] for row in points.values()) for points in curves.values())
    native = curves["C0"][1.0]
    assert lo <= native[0] <= hi
    print(f"Common risk interval (%): [{lo:.9f}, {hi:.9f}]")
    print("Chronos native (risk %, Mbps, narrow seconds):", native)
    for key in ("G1", "C0"):
        result, policy = best_at(curves[key], native[0])
        print(key, policy, result)
        expected = (1227.442075, 28.703288) if key == "G1" else (1314.007246, 64.732255)
        assert all(abs(x-y) < 1e-6 for x, y in zip(result[1:], expected))
    assert abs(native[0] - 4.505208333) < 1e-8
    assert abs(native[1] - 1064.715647) < 1e-6 and abs(native[2] - 24.872250) < 1e-6
