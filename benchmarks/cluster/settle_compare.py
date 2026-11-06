"""Settle-time distributions of two configurations of one system, side by side: n, median, IQR, min/max, the values,
and a two-sided Mann-Whitney U test (scipy; runs that never settled count as +inf, i.e. rank last); and
per target the selectivity swings of runs scored with them (score.py settle["swing"]: time-weighted mean and
max |RT ln(sel/final)|, time > 0.5 kcal/mol off), median and IQR over the runs, with the same test.

    python benchmarks/cluster/settle_compare.py RESULT SYSTEM BASE_CONFIG CAND_CONFIG
(RESULT: a scored result set under the scoreboard results folder, e.g. one made by bench.py score ... --as RESULT)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

RESULTS = Path("/home/user/astera/data/runs/explorer-team/scoreboard/results")


def runs(result: str, system: str, config: str) -> list:
    agg = json.loads((RESULTS / result / "aggregate.json").read_text())
    by_id = {}
    for run in agg.get("runs", [result]):
        for fp in sorted((RESULTS / run).glob("*/score.json")):
            o = json.loads(fp.read_text())
            if o["system"] == system and o["config"] == config and not o.get("control_of"):
                by_id[o["id"]] = o
    return list(by_id.values())


def settles(rs: list) -> list:
    return [((o.get("settle") or {}).get("settle_s") if o.get("ok") else None) for o in rs]


def swings(rs: list, j: int, key: str) -> list:
    out = []
    for o in rs:
        sw = (o.get("settle") or {}).get("swing") if o.get("ok") else None
        if sw and j < len(sw) and sw[j].get(key) is not None:
            out.append(float(sw[j][key]))
    return out


def _dist(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    q1, q3 = np.percentile(xs, 25), np.percentile(xs, 75)
    return {"n": len(xs), "median": float(np.median(xs)), "iqr": float(q3 - q1), "max": float(max(xs))}


def summary(xs: list) -> dict:
    v = np.array([math.inf if x is None else x for x in xs], dtype=float)
    fin = v[np.isfinite(v)]
    q1, q3 = (np.percentile(fin, 25), np.percentile(fin, 75)) if len(fin) else (None, None)
    return {"n": len(v), "never": int((~np.isfinite(v)).sum()), "median": float(np.median(v)),
            "iqr": None if q1 is None else float(q3 - q1), "min": float(fin.min()) if len(fin) else None,
            "max": float(fin.max()) if len(fin) else None, "values": sorted(round(float(x)) if np.isfinite(x) else "inf"
                                                                           for x in v)}


def main(argv: list) -> int:
    result, system, base, cand = argv[:4]
    ra, rb = runs(result, system, base), runs(result, system, cand)
    a, b = settles(ra), settles(rb)
    fa = [math.inf if x is None else x for x in a]
    fb = [math.inf if x is None else x for x in b]
    u = mannwhitneyu(fa, fb, alternative="two-sided")
    out = {"system": system, "base": {"config": base, **summary(a)}, "candidate": {"config": cand, **summary(b)},
           "mann_whitney_u": float(u.statistic), "p_two_sided": float(u.pvalue)}
    n_t = max([len((o.get("settle") or {}).get("swing") or []) for o in ra + rb] or [0])
    out["swing"] = []
    for j in range(n_t):
        row = {"target": j}
        for key in ("mean_kcal", "max_kcal", "time_off_s"):
            xa, xb = swings(ra, j, key), swings(rb, j, key)
            row[key] = {"base": _dist(xa), "candidate": _dist(xb)}
            if xa and xb:
                row[key]["p_two_sided"] = float(mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
        out["swing"].append(row)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
