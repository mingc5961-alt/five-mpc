"""Read-only checks of the pinned G1 public evidence and optional restored weights.

python3 tools/preflight.py --repo /path/to/five-mpc
python3 tools/preflight.py --repo /path/to/five-mpc --model-root /path/to/models

The optional model root follows evidence/model_sha256.json (G1/ and Chronos/).
The files may be symlinks. This program never loads a checkpoint or simulates a run.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


EDITION = Path("editions/g1-paid-cold-start-20261127/evidence")
REVISION = "96b6e630e02c5fc709aef32e5a655b33ddf2463c"
# Hash parsed, canonical JSON so line endings and indentation do not affect this check.
MODEL_MANIFEST = "b3002b96156952464e239177a0b3b014d9ead60e6f7863bbdae3b3335490b466"
WORLD_MANIFEST = "7ac5bd19d3ab0d3ce5bc15249da830d4f516fc6d133aa0b4fc4aa6558ea80b26"
ANCHOR_WORLD = "2d6ee19463199066323fb21187748993544e8ad5eff1f6d3eb294225dcdd4752"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checked_manifest(path, digest):
    data = read_json(path)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise ValueError(f"{path.name} differs from the manifest for {REVISION[:8]}")
    return data


def close(actual, expected, label, tolerance):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"anchor {label}: expected {expected}, got {actual}")


def check_anchor(path):
    receipt = read_json(path)
    if (receipt["split"], receipt["k"], receipt["arm"], receipt["m"]) != ("confirm", 0, "G1", 0.45):
        raise ValueError("incorrect archived anchor identity")
    if receipt["world"] != ANCHOR_WORLD:
        raise ValueError("archived anchor physical-array fingerprint mismatch")
    events = receipt["events"]
    result = receipt["result"]
    if len(events) != 848 or result["probes_per_user"] != 9000:
        raise ValueError("archived anchor event/probe count mismatch")
    if len(result["bits"]) != 16 or len(result["ledger_s"]) != 4:
        raise ValueError("archived anchor user count or ledger shape mismatch")
    for user, (observed_user, end_s) in enumerate(events[:16]):
        if observed_user != user:
            raise ValueError("archived anchor initialization order mismatch")
        close(end_s, 0.0031 + 0.186 * user, "initial completion time", 1e-8)
    close(result["G_mbps"], 1338.9519466382656, "Mbps", 1e-6)
    close(sum(result["bits"]) / 180e6, result["G_mbps"], "delivered bits", 1e-6)
    close(result["R"], 0.07428472222222222, "R (fraction)", 1e-12)
    close(result["narrow_good_data_s"], 58.9878, "usable narrow-beam time", 1e-6)
    close(sum(result["ledger_s"]), 180.0, "ledger total", 1e-6)
    close(result["ledger_s"][0], len(events) * 0.0031, "sensing cost", 1e-7)


def check_models(root, manifest):
    errors = []
    for name, spec in manifest.items():
        path = root / name
        try:
            if not path.is_file():
                errors.append(f"{name}: missing")
                continue
            if path.stat().st_size != spec["bytes"]:
                errors.append(f"{name}: size mismatch")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != spec["sha256"]:
                errors.append(f"{name}: SHA-256 mismatch")
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise ValueError("model bytes failed: " + ", ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="five-mpc repository root")
    parser.add_argument("--model-root", type=Path, help="optional normalized G1/ and Chronos/ tree")
    args = parser.parse_args()
    try:
        evidence = args.repo / EDITION
        models = checked_manifest(evidence / "model_sha256.json", MODEL_MANIFEST)
        worlds = checked_manifest(evidence / "world_hashes.json", WORLD_MANIFEST)
        check_anchor(evidence / "work/factorial_run/run_confirm_0_G1_0.45.json")
        print(f"PASS public evidence for {REVISION[:8]}: {len(models)} model entries, "
              f"{len(worlds)} world-file entries, archived anchor 848 fixes / 2.6288 s / "
              "1338.95195 Mbps / 7.42847% risk / 58.9878 s narrow")
        if args.model_root is not None:
            check_models(args.model_root, models)
            print(f"PASS restored model bytes: {len(models)} sizes and SHA-256 digests")
        else:
            print("SKIP restored model bytes (no --model-root)")
        print("Scientific runtime, world regeneration and live anchor replay remain unchecked.")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
