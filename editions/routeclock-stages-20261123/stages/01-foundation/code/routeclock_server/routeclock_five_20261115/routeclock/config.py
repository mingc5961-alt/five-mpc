from pathlib import Path
import json
import copy

ROOT = Path(__file__).resolve().parents[1]

def load_config(local=False):
    cfg = json.loads((ROOT / "configs/research.json").read_text(encoding="utf-8-sig"))
    if local:
        cfg = copy.deepcopy(cfg)
        cfg["simulation"]["primary_N"] = 3
        cfg["simulation"]["warmup_s"] = 60
        cfg["simulation"]["service_s"] = 12
    return cfg

def dump_json(path, obj):
    import numpy as np
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")
