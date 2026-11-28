"""Run a native G1 development endpoint from the handoff, using a new physical world."""

import argparse
import hashlib
import json
import math
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from experiment_env import REVISION, check_models, configure, stage


NAMESPACE = "g1_icc_server_v1/main/development"
GROUP_COUNTS = (3, 1, 1, 1, 3, 7)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True, help="staged 7/9 runtime")
    parser.add_argument("--out", type=Path, required=True, help="separate development receipt directory")
    parser.add_argument("--world", type=int, required=True, choices=range(1400, 1408))
    parser.add_argument("--multiplier", type=float, required=True, choices=(0.45, 0.60, 0.75, 1.00))
    args = parser.parse_args()
    cpu, runtime, out = (p.resolve() for p in (args.cpu_root, args.runtime, args.out))
    repo = args.repo.resolve()
    source = Path(__file__).resolve().parents[1]
    for original in (repo, source, cpu):
        if out.is_relative_to(original) or runtime.is_relative_to(original):
            parser.error("Choose output and staged runtime outside the repository and downloaded assets")
    if out.is_relative_to(runtime) or runtime.is_relative_to(out):
        parser.error("Use separate sibling directories for output and staged runtime")
    models = check_models(repo, cpu)
    stage(repo, runtime)
    configure(cpu, runtime)
    import numpy as np
    import grurun
    from filelock import FileLock
    from paid_full import GROUPS, World, load_config

    out.mkdir(parents=True, exist_ok=True)
    target = out / f"N01_development_{args.world}_{args.multiplier:.2f}.json"
    with FileLock(str(target) + ".lock", timeout=0):
        if target.exists():
            parser.error(f"Output already exists: {target}. Verify it or use a fresh output directory.")
        cfg = load_config()
        cfg["simulation"].update(warmup_s=0, service_s=180, sensing_budget=3 / 180)
        group_rng = np.random.default_rng(np.random.SeedSequence([20261127, args.world]))
        indices = group_rng.permutation(np.repeat(np.arange(6), GROUP_COUNTS))
        groups = [GROUPS[i] for i in indices]
        world = World(cfg, NAMESPACE, args.world, groups=groups)
        expected = hashlib.sha256(world.positions.tobytes())
        expected.update(np.asarray(world.noise).tobytes())
        expected.update(np.asarray(world.shadows).tobytes())
        with tempfile.TemporaryDirectory(dir=out) as scratch:
            dest = Path(scratch) / "endpoint.json"
            with patch.object(grurun, "world", return_value=world), patch.object(grurun, "path", return_value=dest):
                grurun.run(("development", args.world, "G1", args.multiplier))
            record = json.loads(dest.read_text(encoding="utf-8"))
            r, n = record["result"], len(record["events"])
            if (record["world"] != expected.hexdigest() or r["probes_per_user"] != 9000
                or not math.isclose(sum(r["bits"]) / 180e6, r["G_mbps"], abs_tol=1e-6)
                or not math.isclose(sum(r["ledger_s"]), 180, abs_tol=1e-6)
                or not math.isclose(r["ledger_s"][0], n * 0.0031, abs_tol=1e-7)
                or r["ledger_s"][0] > 3.000001):
                raise ValueError("World fingerprint or delivered-bits/airtime ledger mismatch")
            record["provenance"] = {
                "arm_id": "N01", "protocol": "g1_icc_server_v1", "namespace": NAMESPACE,
                "group_rng_seed": [20261127, args.world], "groups": groups, "source_commit": REVISION,
                "runtime_sha256": {name: digest((runtime / name).read_bytes()) for name in
                                   ("paid_full.py", "grutrain.py", "grudata.py", "grurun.py")},
                "model_sha256": {name: digest(Path(p).read_bytes()) for name, p in models.items()},
                "runner_sha256": digest(Path(__file__).read_bytes()),
                "effective_config": cfg, "radio_threshold_adjustment_db": -5,
                "python": platform.python_version(),
                "packages": {name: version(name) for name in ("numpy","torch","lightgbm","scikit-learn")},
            }
            dest.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            dest.replace(target)
        print(f"{target} | {n} fixes | {r['G_mbps']:.5f} Mbps | risk {100*r['R']:.5f}%")


if __name__ == "__main__":
    main()
