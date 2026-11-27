"""Audit recorded data and regenerate the final CSVs in a fresh temporary directory."""
import json
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parent / "run"


def check(root):
    tasks = json.loads((RUN / "manifest.json").read_bytes())
    status = pd.read_csv(RUN / "live/tasks.csv").set_index("任务ID")
    ids = {t["id"] for t in tasks}
    assert len(ids) == len(tasks) == 2760
    assert status.index.is_unique and set(status.index) == ids and status["状态"].eq("done").all()
    assert json.loads((RUN / "live/status.json").read_bytes())["counts"] == {"done": len(ids)}
    origins = {}
    counts = Counter()
    endpoints = pd.read_csv(RUN / "results/all_main/endpoints.csv").set_index(["a", "w", "b", "m"])
    assert endpoints.index.is_unique and len(endpoints) == 2394
    reused = 0
    for task in tasks:
        key, kind = task["id"], task["kind"]
        args = task["args"]
        path = root / "receipts" / (key + ".json")
        receipt = json.loads(path.read_bytes())
        assert receipt["id"] == key and receipt["args"] == args and set(task["deps"]) <= ids, key
        data = receipt["data"]
        counts[kind] += 1
        if kind == "cell":
            row = data["result"]
            ledger = row["ledger_s"]
            budget = {"T": 3, "M": 4.5, "L": 9}[args["b"]]
            assert abs(sum(ledger) - 180) < 1e-6 and ledger[0] <= budget + .0031001, key
            assert abs(ledger[0] - row["updates"] * .0031) < 1e-6, key
            assert row["probes_per_user"] == 9000 and 0 <= row["unavailable_fraction"] <= row["R"] <= 1, key
            assert len(row["bits"]) == 16 and abs(sum(row["bits"]) / 180e6 - row["G_mbps"]) < 1e-6, key
            if task["batch"] == "R07":
                endpoint = endpoints.loc[args["a"], args["w"], args["b"], args["m"]]
                np.testing.assert_allclose(endpoint[["risk", "mbps", "unavailable", "sense"]].astype(float),
                                           [row["R"] * 100, row["G_mbps"], row["unavailable_fraction"] * 100, ledger[0]])
        if kind == "panel":
            with np.load(path.with_suffix(".npz")) as archive:
                keys, errors = archive["keys"], archive["errors"]
                assert keys.shape == (960, 5) and errors.shape == (960, len(data["names"]), 4), key
                assert list(archive["names"]) == data["names"] and np.isfinite(errors).all(), key
                assert data["n"] == 960 and np.isclose(data["paid_s"], 2.976), key
                np.testing.assert_array_equal(keys, origins.setdefault(args["w"], keys))
                np.testing.assert_allclose(errors.mean(0), data["angles"])
                reused += bool(data.get("reuse"))
    print("Final task counts:", dict(sorted(counts.items())))
    print("Paired panel worlds:", len(origins), "panels with recorded reuse:", reused)
    print("Historical JSON receipts outside the final matrix:", len(list((root / "receipts").glob("*.json"))) - len(ids))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(RUN / "receipts.tar.gz") as archive:
            archive.extractall(root, filter="data")
        check(root)
        for name in ("base.py", "tables.py", "history_report.py", "inference_stats.py", "manifest.json"):
            shutil.copy2(RUN / name, root / name)
        (root / "results").mkdir()
        program = """
import json
from pathlib import Path
from tables import report
for task in json.loads(Path('manifest.json').read_bytes()):
    if task['kind'] == 'tables':
        report(task)
"""
        subprocess.run([sys.executable, "-c", program], cwd=root, check=True)
        compared = 0
        for folder in ("all_main", "core_main", "core_history"):
            original = RUN / "results" / folder
            rebuilt = root / "results" / folder
            files = {p.relative_to(original) for p in original.rglob("*.csv")}
            assert files == {p.relative_to(rebuilt) for p in rebuilt.rglob("*.csv")}
            for rel in sorted(files):
                a, b = pd.read_csv(original / rel), pd.read_csv(rebuilt / rel)
                pd.testing.assert_frame_equal(a, b, rtol=1e-9, atol=1e-9, check_exact=False)
                compared += 1
        print("Rebuilt and matched final CSVs:", compared)


if __name__ == "__main__":
    main()
