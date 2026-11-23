"""Fetch the frozen PX4 list. Default checks file signatures. --full saves logs."""
import argparse
import csv
import hashlib
import os
import time
from pathlib import Path

import requests

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full", action="store_true")
args = parser.parse_args()
session = requests.Session()
# This server can reach the public PX4 CDN directly.
session.trust_env = False
magic = b"ULog\x01\x12\x35"
report = root / ("download.tsv" if args.full else "availability.tsv")
failed = 0
with (root / "audit200.csv").open() as source, report.open("w", buffering=1) as out:
    writer = csv.writer(out, delimiter="\t")
    writer.writerow(["log_id", "status", "bytes_checked", "sha256_full"])
    for row in csv.DictReader(source):
        uid = row["log_id"]
        dest = root / (uid + ".ulg")
        part = dest.with_suffix(".ulg.part")
        try:
            if dest.exists():
                with dest.open("rb") as f:
                    if f.read(7) != magic:
                        raise ValueError("invalid ULog header")
                    f.seek(0)
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                writer.writerow([uid, "cached", dest.stat().st_size, digest])
                continue
            headers = {} if args.full else {"Range": "bytes=0-6"}
            with session.get(row["url"], stream=True, headers=headers, timeout=(10, 45)) as r:
                if r.status_code in (429, 503):
                    writer.writerow([uid, f"rate-limit-{r.status_code}", 0, ""])
                    raise SystemExit("CDN rate limit; retry later")
                r.raise_for_status()
                r.raw.decode_content = True
                prefix = r.raw.read(7)
                if prefix != magic:
                    raise ValueError("invalid ULog header")
                if not args.full:
                    writer.writerow([uid, "header", 7, ""])
                    continue
                with part.open("wb") as f:
                    f.write(prefix)
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                size = part.stat().st_size
                if "Content-Length" in r.headers and size != int(r.headers["Content-Length"]):
                    raise ValueError("incomplete transfer")
                with part.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                os.replace(part, dest)
                writer.writerow([uid, "downloaded", size, digest])
        except (OSError, ValueError, requests.RequestException) as err:
            failed += 1
            writer.writerow([uid, str(err), 0, ""])
        finally:
            time.sleep(0.15)

if failed:
    raise SystemExit(f"{failed} requests failed; inspect {report}")
