"""Publish a verified, self-contained copy of campaign artifacts after completion."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CURRENT = PROJECT / "runs" / "paired600"
ARCHIVES = Path("/data/chen/routeclock_archives")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(source)
    if not target.is_file() or sha(target) != digest:
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise IOError(f"Copy verification failed: {target}")
    return digest


def included(folder):
    for path in sorted(folder.rglob("*")):
        parts = path.relative_to(folder).parts
        if any(p in {"__pycache__", ".pytest_cache", ".git"} or ".work." in p for p in parts):
            continue
        if path.is_file() and path.suffix not in {".lock", ".pid", ".tmp", ".pyc"}:
            yield path


def archive(run, final):
    status = json.loads((run / "status.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if status["stage"] != "complete" or status["confirmation"] != len(manifest["confirmation"]):
        raise ValueError("Campaign has not completed")
    for rows in manifest.values():
        for row in rows:
            if not (Path(row["out"]) / "completed.json").is_file():
                raise ValueError(f"Missing audited completion: {row['out']}")
    for name in ("确认结果.md", "world_metrics.csv", "paired.csv", "chronos_paired.csv", "models/frozen/selection.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(name)
    if final.exists():
        verify(final)
        if sha(final/"runs"/run.name/"manifest.json") != sha(run/"manifest.json"):
            raise ValueError("Archive name is already used by another manifest")
        return
    stage = final.with_name("."+final.name+".partial")
    stage.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for name in ("tools", "tests", "routeclock", "configs", "results"):
        for path in included(PROJECT / name):
            candidates[path.relative_to(PROJECT)] = path
    for path in PROJECT.iterdir():
        if path.is_file(): candidates[Path(path.name)] = path
    for path in included(run):
        candidates[Path("runs") / run.name / path.relative_to(run)] = path
    checkpoints = {Path(r["argv"][r["argv"].index("--checkpoint")+1])
                   for r in manifest["confirmation"] if "--checkpoint" in r["argv"]}
    for ckpt in checkpoints:
        for path in included(ckpt):
            candidates[Path("checkpoints")/ckpt.name/path.relative_to(ckpt)] = path
    hashes = {str(name):copy_verified(source,stage/name) for name,source in sorted(candidates.items())}
    (stage/"SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name,h in hashes.items()))
    verify(stage)
    stage.rename(final)


def verify(folder):
    lines=(folder/"SHA256SUMS").read_text().splitlines()
    if not lines:raise ValueError("Empty checksum manifest")
    for line in lines:
        digest,name=line.split("  ",1)
        if sha(folder/name)!=digest:raise IOError(f"Checksum mismatch: {name}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=CURRENT)
    p.add_argument("--verify",type=Path)
    a=p.parse_args()
    if a.verify:
        verify(a.verify)
        return
    root=a.root.resolve()
    final=ARCHIVES/root.name
    ARCHIVES.mkdir(parents=True,exist_ok=True)
    status_file=ARCHIVES/(root.name+"_归档状态.json")
    def state(phase,**details):
        tmp=status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(phase=phase,source=str(root),destination=str(final),updated=time.time(),**details),ensure_ascii=False,indent=2))
        os.replace(tmp,status_file)
    with (ARCHIVES/("."+root.name+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state("等待实验完成")
        while json.loads((root/"status.json").read_text()).get("stage")!="complete":
            time.sleep(30)
        for attempt in range(1,4):
            try:
                state("正在归档并核验",attempt=attempt)
                archive(root,final)
                state("已完成 SHA-256 核验")
                return
            except (OSError,ValueError) as e:
                state("归档异常",attempt=attempt,error=str(e))
                if attempt==3:raise
                time.sleep(30)


if __name__=="__main__":main()
