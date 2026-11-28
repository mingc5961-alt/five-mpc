"""Stage the pinned 7/9 scripts and configure a local copy of the CPU archive."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path, PurePosixPath


REVISION = "96b6e630e02c5fc709aef32e5a655b33ddf2463c"
INDEX_SHA256 = "fd00b0368b73f63fc248fcb0726f3bd8d8891d98c4114d73898833afefe6dcd0"
EDITION = "editions/g1-paid-cold-start-20261127/evidence"
PATCH = Path(__file__).resolve().parents[1]
FUSION = Path("experiment/gate/fusion")
G1_WEIGHTS = FUSION / "neural/gru_8_residual"
MODEL_PATHS = {
    "G1/best.pt": G1_WEIGHTS / "best.pt",
    "G1/fusion.json": G1_WEIGHTS / "fusion.json",
    "G1/frozen_gate.json": FUSION / "frozen_gate.json",
    "G1/tube.pkl": FUSION / "closed/tubes/G1.pkl",
    "Chronos/best.pt": Path("confirm_0db/training/Chronos-2/7/best.pt"),
    "Chronos/tube.pkl": Path("confirm_0db/tubes/Chronos-2/7/bundle.pkl"),
}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE).stdout


@lru_cache(maxsize=4)
def source_index(repo):
    path = repo / "index.json"
    if not path.is_file() or (repo / ".git").exists():
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != INDEX_SHA256:
        raise ValueError(f"Bundled pinned-reference index changed: {path}")
    return json.loads(data)


def pinned_bytes(repo, relative):
    """Read one byte-exact git blob from the pinned checkout or verified snapshot."""
    repo = Path(repo).resolve()
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected repository-relative path: {relative}")
    index = source_index(repo)
    if index is None:
        return git(repo, "show", f"{REVISION}:{relative.as_posix()}")
    expected = index.get(relative.as_posix())
    if expected is None:
        raise FileNotFoundError(f"File absent from the pinned snapshot: {relative}")
    payload = (repo / relative).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Changed pinned-reference file: {relative}")
    return payload


def check_models(repo, cpu_root, with_chronos=False):
    manifest = json.loads(pinned_bytes(repo, f"{EDITION}/model_sha256.json"))
    if set(manifest) != set(MODEL_PATHS):
        raise ValueError("Unexpected set of published model files")
    paths = {name: cpu_root / relative for name, relative in MODEL_PATHS.items()
             if with_chronos or name.startswith("G1/")}
    for name, path in paths.items():
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != manifest[name]["bytes"] or digest != manifest[name]["sha256"]:
            raise ValueError(f"Published model fingerprint mismatch at {path}")
    if with_chronos:
        base_weights = cpu_root / "chronos_snapshot/model.safetensors"
        with base_weights.open("rb") as handle:
            base_digest = hashlib.file_digest(handle, "sha256").hexdigest()
        # The original CPU archive's manifest pins this backbone separately from the six model entries.
        if base_digest != "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42":
            raise ValueError(f"Original Chronos backbone fingerprint mismatch at {base_weights}")
    return {name: str(path) for name, path in paths.items()}


def configure(cpu_root, runtime):
    """Set process-local paths before importing any of the scientific modules."""
    cpu_root, runtime = Path(cpu_root).resolve(), Path(runtime).resolve()
    closed = cpu_root / FUSION / "closed"
    base, snap = cpu_root / "confirm_0db", cpu_root / "chronos_snapshot"
    for path in (runtime / "paid_full.py", runtime / "grutrain.py", runtime / "grudata.py",
                 closed / "simulate.py", closed / "common.py", closed.parent / "neural.py",
                 base / "work/code/routeclock_refine/routeclock/world.py",
                 base / "work/baseline_cache/classical.pkl", snap / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing runtime input: {path}")
    env = {"PAID_SOURCE": str(closed), "PAID_OUTPUT": str(runtime / "paid_full"),
           "ROUTECLOCK_BASE": str(base), "CHRONOS_SNAPSHOT": str(snap),
           "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    os.environ.update(env)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime))
    return env


def stage(repo, runtime):
    repo = Path(repo).resolve()
    prefix = f"{EDITION}/work/"
    index = source_index(repo)
    if index is None:
        entries = git(repo, "ls-tree", "--name-only", f"{REVISION}:{prefix[:-1]}")
        names = sorted(n for n in entries.decode().splitlines() if n.endswith(".py"))
    else:
        names = sorted(n[len(prefix):] for n in index if n.startswith(prefix)
                       and "/" not in n[len(prefix):] and n.endswith(".py"))
    required = {"grurun.py", "grudata.py", "factorial_data.py", "factorial_closed.py"}
    if not required.issubset(names):
        raise ValueError("Missing scripts in pinned git tree")
    files = {name: pinned_bytes(repo, prefix + name) for name in names}
    for name in ("paid_full.py", "grutrain.py"):
        files[name] = (PATCH / "runtime" / name).read_bytes()
    if runtime.exists():
        for name, content in files.items():
            if (runtime / name).read_bytes() != content:
                raise ValueError(f"Existing stage differs at {runtime / name}; choose a fresh path")
        return
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime.parent) as directory:
        tmp = Path(directory) / "stage"
        tmp.mkdir()
        for name, content in files.items():
            (tmp / name).write_bytes(content)
        tmp.rename(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="five-mpc checkout containing the pinned commit, or the packaged reference directory")
    parser.add_argument("--cpu-root", required=True, type=Path, help="extracted gru4505 directory")
    parser.add_argument("--runtime", required=True, type=Path, help="dedicated writable stage for current research code")
    parser.add_argument("--with-chronos", action="store_true", help="also require the full Chronos weights and compare its two archived files to 7/9 fingerprints")
    args = parser.parse_args()
    cpu_root, runtime = args.cpu_root.resolve(), args.runtime.resolve()
    for original in (args.repo.resolve() / EDITION, cpu_root, PATCH / "runtime"):
        if runtime == original or runtime.is_relative_to(original):
            parser.error("--runtime must be separate from source evidence, recovered scripts and CPU archives")
    models = check_models(args.repo, cpu_root, args.with_chronos)
    stage(args.repo, runtime)
    paths = configure(cpu_root, runtime)
    print(json.dumps({"revision": REVISION, "runtime": str(runtime), "environment": paths,
                      "verified_models": models}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
