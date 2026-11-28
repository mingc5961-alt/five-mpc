"""List the core workload, or export its development coordinates after quota selection."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocol.json")
    parser.add_argument("--dev-csv", type=Path, help="new output file; includes N01 anchor coordinates")
    parser.add_argument("--quota", type=int, help="actual K from development")
    parser.add_argument("--pm-period-ms", type=int, help="actual fixed period from development")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    quota = protocol["quota"]
    horizon = protocol["anchor"]
    limit = round(horizon["sensing_budget_s"] * 1_000_000) // horizon["fix_cost_us"]
    if args.dev_csv is not None:
        if args.quota is None or not horizon["users"] <= args.quota <= limit:
            parser.error("--dev-csv needs a feasible development-selected --quota")
        if args.pm_period_ms is None or args.pm_period_ms < 3000:
            parser.error("--dev-csv needs the development-selected --pm-period-ms (>=3000 for B=3)")
    splits = protocol["split_id_ranges_inclusive"]
    first, last = splits["development"]
    worlds = range(first, last + 1)
    confirm_n = splits["confirmation"][1] - splits["confirmation"][0] + 1
    scales = protocol["working_points"]["multipliers"]
    cases = []
    counts = Counter()
    seen = set()
    for row_id, clock, predictor, refresh, replicas in protocol["core_matrix"]:
        if row_id in seen or replicas not in ("dqn", "single"):
            raise ValueError(f"Invalid matrix row {row_id}")
        seen.add(row_id)
        seeds = quota["dqn_seeds"] if replicas == "dqn" else ["NA"]
        counts[row_id[0]] += len(seeds)
        for seed in seeds:
            for world in worlds:
                for multiplier in scales:
                    count = args.quota if clock in ("fixed_success", "fixed_count") else ""
                    period = {"period_from_dev": args.pm_period_ms, "period_3s": 3000}.get(refresh, "")
                    cases.append((row_id, "main", "development", world, clock, predictor,
                                  refresh, seed, count, period, multiplier))
    for a, b in protocol["primary_closed_contrasts"] + protocol["secondary_contrasts"]:
        if not {a, b}.issubset(seen):
            raise ValueError(f"Unknown contrast {a}, {b}")
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate development coordinates")
    total = sum(counts.values())
    print(f"Rows: {len(seen)}; policy instances: {total} {dict(counts)}")
    print(f"Development: {len(cases)}; confirmation upper bound: {2 * confirm_n * total}")
    if args.dev_csv is not None:
        with args.dev_csv.open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("row_id", "scenario", "split", "world", "clock", "predictor",
                             "refresh", "scheduler_seed", "quota", "period_ms", "multiplier"))
            writer.writerows(cases)
        print(f"Wrote {args.dev_csv} (coordinates only, not a frozen execution manifest)")


if __name__ == "__main__":
    main()
