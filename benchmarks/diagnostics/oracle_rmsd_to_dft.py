"""RMSD of each oracle's transition state to the DFT-optimized one (all atoms, best proper rotation, same atom order).

    python oracle_rmsd_to_dft.py SPEC.json OUT.json

SPEC: [{"key": "system:label", "dft": xyz file (first frame) or [multi-frame xyz, frame title],
        "oracles": {"gxtb": xyz, "gfn2": xyz, "uma": xyz}}, ...]   paths relative to the spec
OUT: {key: {oracle: rmsd_A}} and, per oracle, the median and maximum over the keys.
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np


def frames(path):
    L, out, i = Path(path).read_text().splitlines(), {}, 0
    while i < len(L):
        n = int(L[i])
        out.setdefault(L[i + 1].strip(), np.array([[float(v) for v in l.split()[1:4]] for l in L[i + 2:i + 2 + n]]))
        out.setdefault(0, out[L[i + 1].strip()])
        i += n + 2
    return out


def rmsd(X, Y) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((X @ R.T) - Y) ** 2).sum() / len(X)))


def main(spec_path, out_path):
    base = Path(spec_path).resolve().parent
    out = {}
    for item in json.loads(Path(spec_path).read_text()):
        d = item["dft"]
        D = frames(base / d[0])[d[1]] if isinstance(d, list) else frames(base / d)[0]
        out[item["key"]] = {k: rmsd(frames(base / f)[0], D) for k, f in item["oracles"].items()}
    ks = sorted({k for v in out.values() for k in v})
    out["summary"] = {k: {"median_A": statistics.median(v[k] for v in out.values() if k in v),
                          "max_A": max(v[k] for v in out.values() if k in v)} for k in ks}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
