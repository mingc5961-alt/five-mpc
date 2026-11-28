"""Reference clock for a G1 matched-cost simulation and audit of its public prefix panel.

    python aoi_reference.py
    python aoi_reference.py --panel path/to/chronos_panel.csv --csv pilot.csv

Only standard-library Python is needed. The clock returns opportunities in microseconds.
The real runner must still debit its TokenBucket and DeferredLedger at the chosen times.
"""

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class QuotaGrid:
    service_us: int = 180_000_000
    budget_us: int = 3_000_000
    fix_us: int = 3_100
    users: int = 16
    quota: int = 854

    def __post_init__(self):
        if not 0 < self.fix_us <= self.budget_us <= self.service_us:
            raise ValueError("Require 0 < fix <= budget <= service")
        if not 1 <= self.users <= self.quota <= self.opportunities:
            raise ValueError("Quota must cover initialization and fit within the ledger")
        if self.start_us(self.opportunities - 1) + self.fix_us > self.service_us:
            raise ValueError("The last scheduled fix cannot finish before service ends")

    @property
    def opportunities(self):
        return self.budget_us // self.fix_us

    def start_us(self, index):
        if not 0 <= index < self.opportunities:
            raise IndexError(index)
        # Round up the interval, so adjacent opportunities always respect the token rate.
        spacing_us = math.ceil(Fraction(self.fix_us * self.service_us, self.budget_us))
        return index * spacing_us

    def allowed(self, index, used):
        """Return (may_wait, may_sense). Index and remaining counts include this tick."""
        if not 0 <= index < self.opportunities or not 0 <= used <= min(index, self.quota):
            raise ValueError("Invalid index or cumulative count")
        if used < min(index, self.users):
            raise ValueError("Initialization or earlier decisions violated the quota")
        ticks_left = self.opportunities - index
        required = self.quota - used
        if required > ticks_left:
            raise ValueError("The quota is already unreachable")
        if index < self.users:
            return False, True
        return required < ticks_left, required > 0

    def fixed_indices(self):
        """Fixed clock for a recipient-only experiment, including all first fixes."""
        extra = self.quota - self.users
        room = self.opportunities - self.users
        indices = list(range(self.users))
        for j in range(extra):
            indices.append(self.users + ((2 * j + 1) * room) // (2 * extra))
        return tuple(indices)


def verify_grid(grid):
    """Check a fixed, early-spending and late-spending policy against identical masks."""
    ticks = range(grid.opportunities)
    fixed = grid.fixed_indices()
    assert len(fixed) == len(set(fixed)) == grid.quota
    assert fixed[: grid.users] == tuple(range(grid.users))
    starts = [grid.start_us(i) for i in ticks]
    assert all((b - a) * grid.budget_us >= grid.fix_us * grid.service_us
               for a, b in zip(starts, starts[1:]))
    for preferred in (set(fixed), set(ticks), set()):
        used = 0
        for i in ticks:
            wait, sense = grid.allowed(i, used)
            if not wait or (sense and i in preferred):
                assert sense
                used += 1
        assert used == grid.quota and used * grid.fix_us <= grid.budget_us
    return len(ticks), grid.quota * grid.fix_us / 1e6


AGES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0)


def summarize_public_panel(source, destination):
    """Paired stale-forecast error at h=0, from the original exogenous diagnostic only."""
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    pairs = defaultdict(dict)
    with source.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            arm = row['arm']
            if arm not in ('G1', 'C'):
                continue
            key = (int(row['world']), int(row['u']), int(row['n']))
            if arm in pairs[key]:
                raise ValueError(f'Duplicate entry {key} {arm}')
            angles = tuple(float(row[str(age)]) for age in AGES)
            if not all(math.isfinite(x) and x >= 0 for x in angles):
                raise ValueError(f'Invalid angle error {key} {arm}')
            pairs[key][arm] = angles
    counts = (1, 2, 3, 4, 6, 8, 12, 24, 40)
    expected = {(w, u, n) for w in range(8) for u in range(16) for n in counts}
    if set(pairs) != expected or any(set(v) != {'G1', 'C'} for v in pairs.values()):
        raise ValueError('Incomplete 8-world, 16-user, 9-history paired public panel')
    with destination.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('fixes_at_issue', 'age_s', 'extra_horizon_s', 'worlds', 'paired_anchors',
                         'g1_angular_error_deg', 'chronos_angular_error_deg',
                         'delta_g1_minus_chronos_deg', 'worlds_with_lower_g1_error'))
        for n in counts:
            for idx, age in enumerate(AGES):
                world_g1 = []
                world_c = []
                for w in range(8):
                    rows = [pairs[w, u, n] for u in range(16)]
                    world_g1.append(mean(row['G1'][idx] for row in rows))
                    world_c.append(mean(row['C'][idx] for row in rows))
                diff = [a - b for a, b in zip(world_g1, world_c)]
                writer.writerow((n, age, 0, 8, 128, mean(world_g1), mean(world_c), mean(diff),
                                 sum(x < 0 for x in diff)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quota', type=int, default=854)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    if (args.panel is None) != (args.csv is None):
        parser.error('Use --panel and --csv together')
    grid = QuotaGrid(quota=args.quota)
    ticks, cost = verify_grid(grid)
    print(f'{ticks} feasible opportunities, {grid.users} initial fixes, '
          f'{grid.quota} paid fixes, {cost:.4f} s sensing')
    if args.panel is not None:
        summarize_public_panel(args.panel, args.csv)
        print(f'Wrote {9 * len(AGES)} paired age/history cells to {args.csv}')


if __name__ == '__main__':
    main()
