"""Run the historical G1 or Chronos confirm world 0 afresh and compare receipts."""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch


EDITION = Path("editions/g1-paid-cold-start-20261127/evidence/work")
ANCHOR_WORLD = "2d6ee19463199066323fb21187748993544e8ad5eff1f6d3eb294225dcdd4752"
REVISION = "96b6e630e02c5fc709aef32e5a655b33ddf2463c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_bytes(repo, relative):
    from experiment_env import pinned_bytes
    return pinned_bytes(repo, (EDITION / relative).as_posix())


def verify(receipt, reference):
    problems = []
    def check(condition, label):
        if not condition:
            problems.append(label)

    for key in ("world", "split", "k", "arm", "m"):
        check(receipt[key] == reference[key], key)
    check(receipt["world"] == ANCHOR_WORLD, "world physical-array SHA-256")
    actual, expected = receipt["events"], reference["events"]
    check(len(actual) == len(expected), "event count")
    for i, ((u, t), (v, s)) in enumerate(zip(actual, expected)):
        if u != v or not math.isclose(t, s, abs_tol=1e-6, rel_tol=0):
            problems.append(f"first event divergence at {i}: actual={(u,t)}, reference={(v,s)}")
            break
    r, ref = receipt["result"], reference["result"]
    for field, atol in (("G_mbps", .001), ("R", 1e-10), ("unavailable_fraction", 1e-10),
                        ("narrow_good_data_s", 1e-6), ("probes_per_user", 0)):
        check(math.isclose(r[field], ref[field], abs_tol=atol, rel_tol=0), field)
    for field, atol in (("ledger_s", 1e-6), ("bits", 100.0)):
        check(len(r[field]) == len(ref[field]), field + " length")
        for i, (a, b) in enumerate(zip(r[field], ref[field])):
            check(math.isclose(a, b, rel_tol=0, abs_tol=atol), f"{field}[{i}]")
    check(math.isclose(sum(r["bits"])/180e6, r["G_mbps"], rel_tol=0, abs_tol=1e-6),
          "bits/elapsed time")
    check(math.isclose(sum(r["ledger_s"]), 180, rel_tol=0, abs_tol=1e-6), "total ledger")
    check(math.isclose(r["ledger_s"][0], len(actual)*.0031, rel_tol=0, abs_tol=1e-6),
          "paid fixes ledger")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="five-mpc checkout or bundled pinned reference directory")
    parser.add_argument("--runtime", required=True, type=Path, help="restored paid_full.py and grutrain.py directory")
    parser.add_argument("--cpu-root", type=Path, help="portable extracted gru4505 directory; omit for original configured installation")
    parser.add_argument("--out", required=True, type=Path, help="new unique output directory; existing directories are rejected")
    parser.add_argument("--arm", choices=("G1","C"), default="G1")
    parser.add_argument("--scale", type=float, help="default G1 0.45, Chronos 0.75")
    args = parser.parse_args()
    scale = args.scale if args.scale is not None else (.45 if args.arm == "G1" else .75)
    archive = args.repo.resolve() / EDITION
    runtime = args.runtime.resolve()
    out = args.out.resolve()
    sources = [runtime / name for name in ("paid_full.py", "grutrain.py")]
    reference = archive / "factorial_run" / f"run_confirm_0_{args.arm}_{scale:.2f}.json"
    for item in sources:
        if not item.is_file():
            parser.error(f"Required file missing: {item}")
    reference_data = archive_bytes(args.repo, reference.relative_to(archive))
    scripts = {name: archive_bytes(args.repo, name) for name in ("grurun.py", "grudata.py", "factorial_data.py")}
    if any(out.is_relative_to(p) for p in (archive, runtime)):
        parser.error("--out must be outside the scientific inputs")
    out.parent.mkdir(exist_ok=True, parents=True)
    out.mkdir(exist_ok=False)
    stage = out / "work"
    stage.mkdir()
    for name, content in scripts.items():
        (stage / name).write_bytes(content)
    for source in sources:
        shutil.copyfile(source, stage / source.name)
    sys.dont_write_bytecode = True
    if args.cpu_root:
        from experiment_env import check_models, configure
        check_models(args.repo, args.cpu_root.resolve(), with_chronos=args.arm == "C")
        configure(args.cpu_root.resolve(), stage)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["PAID_OUTPUT"] = str(out)
    sys.path.insert(0, str(stage))
    start = time.monotonic()
    print("Loading scientific modules for a fresh world-0 replay.", flush=True)
    import grurun
    import factorial_data
    fresh = out / reference.name
    print(f"Regenerating confirm world 0 and running {args.arm} at scale {scale}.", flush=True)
    with patch.object(grurun, "world", factorial_data.fresh_world), patch.object(grurun, "path", lambda *job: fresh):
        grurun.run(("confirm", 0, args.arm, scale))
    elapsed = time.monotonic() - start
    receipt = json.loads(fresh.read_text(encoding="utf-8"))
    problems = verify(receipt, json.loads(reference_data))
    modules = ("grurun", "grudata", "grutrain", "paid_full", "factorial_data", "simulate", "common",
               "neural", "routeclock.world", "runtime.fast_ledger")
    audit = dict(status="FAIL" if problems else "PASS", wall_s=elapsed, python=sys.version,
                 python_executable=sys.executable, source_mode="portable" if args.cpu_root else "native",
                 scientific_files={name: {"name": Path(sys.modules[name].__file__).name,
                                          "sha256":sha(Path(sys.modules[name].__file__))}
                                   for name in modules if name in sys.modules},
                 scale=scale, arm=args.arm, scenario="confirm", world=0, world_sha256=receipt["world"],
                 events=len(receipt["events"]), G_mbps=receipt["result"]["G_mbps"],
                 risk_fraction=receipt["result"]["R"], sensing_s=receipt["result"]["ledger_s"][0],
                 narrow_s=receipt["result"]["narrow_good_data_s"], reference_revision=REVISION,
                 reference_sha256=hashlib.sha256(reference_data).hexdigest(),
                 replay_sha256=sha(fresh), differences=problems)
    (out/"verification.json").write_text(json.dumps(audit,indent=2)+"\n", encoding="utf-8")
    print(f"{audit['status']} {args.arm}: {len(receipt['events'])} fixes, {audit['G_mbps']:.5f} Mbps, {elapsed:.1f}s; verification.json", flush=True)
    for difference in problems:
        print(difference, file=sys.stderr)
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
