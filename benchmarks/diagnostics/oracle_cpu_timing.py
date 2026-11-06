"""Reference cost of an oracle on one CPU core: per TS, one energy+gradient, one Hessian, and the TS search + IRC from
the g-xTB TS, for UMA (device cpu, torch on 1 thread) -- the same work oracle.py times on the GPU.

    python oracle_cpu_timing.py OUT.json SYSTEM_DIR:TS_LABEL [...]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv):
    import torch

    torch.set_num_threads(1)
    import oracle
    from explorer_level import load

    oracle.ORACLE.update(name="uma", device="cpu")
    out = {}
    for item in argv[1:]:
        sysdir, label = item.split(":")
        sysdir = Path(sysdir)
        spec = json.loads((sysdir / "ceiling_spec.json").read_text())
        st = next(s for s in spec["steps"] if s["label"] == label)
        g = load(sysdir / st["xyz"])
        tmp = sysdir / "oracle_uma_cpu"
        tmp.mkdir(exist_ok=True)
        eng = oracle.oracle_engine(tmp)
        t0 = time.time()
        rec = {"call": oracle.per_call(g, eng)}
        rec["search"] = oracle.search(g, eng, tmp, label)
        rec["wall_s"] = time.time() - t0
        out[item] = rec
        print(item, json.dumps({k: v for k, v in rec.items() if k != "search"}), rec["search"].get("t_total"), flush=True)
        Path(argv[0]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
