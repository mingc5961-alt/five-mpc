"""Recreate audit200.csv from the retained PX4 metadata snapshot."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
groups = defaultdict(list)
seen = set()
multirotors = {"Quadrotor", "Hexarotor", "Octorotor", "Tricopter"}
with gzip.open(root / "dbinfo.json.gz", "rt") as source:
    for entry in json.load(source):
        year = entry["log_date"][:4]
        uid = entry["log_id"]
        if year not in {"2024", "2025"} or entry["mav_type"] not in multirotors or uid in seen:
            continue
        seen.add(uid)
        durations = Counter()
        for mode, seconds in entry.get("flight_mode_durations") or []:
            durations[mode] += seconds
        dominant = min(durations, key=lambda m: (-durations[m], m)) if durations else "unknown"
        groups[(year, str(dominant))].append(entry)

population = len(seen)
slots = {key: 200 * len(entries) // population for key, entries in groups.items()}
by_remainder = sorted(groups, key=lambda k: (-(200 * len(groups[k]) % population), k))
for key in by_remainder[:200 - sum(slots.values())]:
    slots[key] += 1

def draw_rank(entry):
    return hashlib.sha256(("271828:" + entry["log_id"]).encode()).digest()

with (root / "audit200.csv").open("w", newline="") as target:
    writer = csv.writer(target)
    writer.writerow(["index_year", "dominant_mode", "N", "draw_n", "log_id", "url"])
    for key, entries in sorted(groups.items()):
        for entry in sorted(entries, key=draw_rank)[:slots[key]]:
            writer.writerow([*key, len(entries), slots[key], entry["log_id"], entry["download_url"]])
