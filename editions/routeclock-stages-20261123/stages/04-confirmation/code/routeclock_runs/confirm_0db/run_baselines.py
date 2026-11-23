from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from runtime import remaining_runs as runtime
from refine import world_cache

ROOT = Path(__file__).resolve().parent
NAMESPACE = "confirm_0db_20261122/v1"
WORLDS = range(40)
BUDGETS = ("T", "M", "L")
METHODS = ("F", "I", "RC")
WORKERS = 6

def write_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, default=lambda value: value.tolist(), sort_keys=True))
    temp.replace(path)

def paired_world(config, namespace, world):
    return world_cache.paired_world(config, NAMESPACE, world)

def run_world(world):
    runtime.OUT = ROOT / "work" / "baseline_cache"
    world_cache.BASE = ROOT / "work" / "baseline_cache"
    runtime.network.paired_world = paired_world
    runner = runtime.Runner()
    for budget in BUDGETS:
        for method in METHODS:
            row = dict(model=method, phase="confirm", world=world, condition=budget, mcs_margin_db=0, tube="default", risk_scale=1, max_interval=10)
            if method == "F":
                row["fixed"] = {"pair": [8, 32], "period": 3.0}
            config = dict(row, namespace=NAMESPACE, inputs=INPUT_HASHES)
            key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            folder = ROOT / "results" / "baselines" / f"{world:03d}_{budget}_{method}"
            folder.mkdir(parents=True, exist_ok=True)
            status = folder / "complete.json"
            if status.exists() and json.loads(status.read_text())["config_sha256"] == key:
                continue
            start = time.monotonic()
            payload = runner.execute(row)
            write_json(status, dict(config_sha256=key, frozen_config=config, elapsed_s=time.monotonic()-start, result=payload))
            print(world, budget, method, round(time.monotonic()-start, 1), flush=True)
    return world

def file_hash(file):
    with file.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

if __name__ == "__main__":
    cache = ROOT / "work" / "baseline_cache"
    cache.mkdir(exist_ok=True)
    archive = ROOT.parent / "refine_v1" / "remaining_queue"
    for name in ("F.json", "classical.pkl"):
        link = cache / name
        if not link.exists():
            link.symlink_to(archive / name)
    code = ROOT / "work" / "code" / "routeclock_refine"
    files = [p for p in code.rglob("*.py") if p.is_file()] + [code / "configs" / "research.json", cache / "F.json", cache / "classical.pkl", Path(__file__)]
    INPUT_HASHES = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}
    manifest = ROOT / "baseline_manifest.json"
    config = dict(namespace=NAMESPACE, worlds=list(WORLDS), budgets=BUDGETS, methods=METHODS, mcs_margin_db=0, inputs=INPUT_HASHES)
    if manifest.exists():
        assert json.loads(manifest.read_text()) == json.loads(json.dumps(config)), "Frozen inputs changed"
    else:
        write_json(manifest, config)
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(run_world, WORLDS))
