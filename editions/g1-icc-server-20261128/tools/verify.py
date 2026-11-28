"""Verify the archive using only the Python standard library."""
import csv
import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "results/paired_linux_v2"
COUNTS = {"development": 1504, "confirmation": 2352, "multiplier-extension": 656,
          "real-motion": 160, "preliminary-linux": 1384}


def require(ok, message):
    if not ok:
        raise ValueError(str(message))


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right, subject):
    require(math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-6), subject)


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def by_member(path):
    rows = read_table(path)
    index = {(r["archive"], r["member"]): r for r in rows}
    require(len(rows) == len(index), path)
    return index


def verify_files():
    seen = set()
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(name not in seen and file_digest(ROOT / name) == digest, name)
        seen.add(name)
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
              and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    require(actual == seen, "version file set")
    return len(seen)


def verify_receipts():
    members = by_member(ROOT / "indexes/archive-members.csv")
    endpoints = by_member(ROOT / "indexes/endpoints.csv")
    counts = Counter()
    open_runs = defaultdict(list)
    seen = set()
    for path in sorted((ROOT / "archives").glob("*.tar.gz")):
        archive = path.name.removesuffix(".tar.gz")
        with tarfile.open(path, "r|gz") as tar:
            for member in tar:
                key = (archive, member.name)
                require(member.isfile() and key in members and key not in seen, key)
                content = tar.extractfile(member).read()
                digest = hashlib.sha256(content).hexdigest()
                require(digest == members[key]["sha256"] and len(content) == int(members[key]["bytes"]), key)
                seen.add(key)
                if archive not in COUNTS:
                    continue
                index = endpoints[key]
                row = json.loads(content)
                result = row["result"]
                seconds = row.get("duration", 180)
                require(digest == index["sha256"] and len(row["events"]) == int(index["paid_fixes"]), key)
                close(sum(result["ledger_s"]), seconds, key)
                close(result["ledger_s"][0], index["sensing_s"], key)
                close(sum(result["bits"]) / seconds / 1e6, result["G_mbps"], key)
                close(result["G_mbps"], index["goodput_mbps"], key)
                close(100 * result["R"], index["risk_pct"], key)
                counts[archive] += 1
                if archive == "real-motion":
                    open_runs[(row["source"], row["view"], row["panel"], row["row"])].append(result)
    require(seen == set(members) and set(endpoints) <= seen and counts == COUNTS, counts)
    require(len(endpoints) == sum(COUNTS.values()), len(endpoints))
    return open_runs, len(members)


def verify_field(open_runs):
    panels = {}
    for (source, view, panel, row), runs in open_runs.items():
        panels[source, view, panel, row] = (mean(r["G_mbps"] for r in runs), mean(100*r["R"] for r in runs))
    contrasts = read_table(PAIRED / "field_extension/paired_throughput.csv")
    require(len(panels) == 80 and len(contrasts) == 48, "field panel dimensions")
    for r in contrasts:
        def values(arm):
            return {p: stats for (s, v, p, row), stats in panels.items()
                    if (s, v, row) == (r["source"], r["view"], arm)}
        a, b = values(r["first"]), values(r["second"])
        require(a.keys() == b.keys() and len(a) == int(r["assemblies"]), r)
        first, second = mean(v[0] for v in a.values()), mean(v[0] for v in b.values())
        close(first, r["first_mbps"], r)
        close(second, r["second_mbps"], r)
        close(first-second, r["gain_mbps"], r)
        close(100*(first-second)/second, r["gain_pct"], r)
        close(mean(a[p][1]-b[p][1] for p in a), r["risk_difference_pp"], r)


if __name__ == "__main__":
    file_count = verify_files()
    field, member_count = verify_receipts()
    verify_field(field)
    print(f"OK: {file_count} files, {member_count} archive members, {sum(COUNTS.values())} endpoints, 48 field contrasts")
